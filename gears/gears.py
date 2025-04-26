from copy import deepcopy
import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import pandas as pd



from .model import GEARS_Model
from .inference import evaluate, compute_metrics, deeper_analysis, \
                  non_dropout_analysis
from .utils import loss_fct, uncertainty_loss_fct, parse_any_pert, \
                  get_similarity_network, print_sys, GeneSimNetwork, \
                  create_cell_graph_dataset_for_prediction, get_mean_control, \
                  get_GI_genes_idx, get_GI_params

torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

class GEARS:
    """
    GEARS base model class
    """

    def __init__(self, pert_data, 
                 device = 'cuda',
                 weight_bias_track = False, 
                 proj_name = 'GEARS', 
                 exp_name = 'GEARS'):
        """
        Initialize GEARS model

        Parameters
        ----------
        pert_data: PertData object
            dataloader for perturbation data
        device: str
            Device to run the model on. Default: 'cuda'
        weight_bias_track: bool
            Whether to track performance on wandb. Default: False
        proj_name: str
            Project name for wandb. Default: 'GEARS'
        exp_name: str
            Experiment name for wandb. Default: 'GEARS'

        Returns
        -------
        None

        """

        self.weight_bias_track = weight_bias_track
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = device
        self.config = None
        
        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.default_pert_graph = pert_data.default_pert_graph
        self.saved_pred = {}
        self.saved_logvar_sum = {}
        
        self.ctrl_expression = torch.tensor(
            np.mean(self.adata.X[self.adata.obs.condition == 'ctrl'],
                    axis=0)).reshape(-1, ).to(self.device)
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {pert_full_id2pert[i]: j for i, j in
                            self.adata.uns['non_zeros_gene_idx'].items() if
                            i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        
        gene_dict = {g:i for i,g in enumerate(self.gene_list)}
        self.pert2gene = {p: gene_dict[pert] for p, pert in
                          enumerate(self.pert_list) if pert in self.gene_list}

    def tunable_parameters(self):
        """
        Return the tunable parameters of the model

        Returns
        -------
        dict
            Tunable parameters of the model

        """

        return {'hidden_size': 'hidden dimension, default 64',
                'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
                'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
                'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
                'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
                'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co expression graph, default 20',
                'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
                'uncertainty': 'whether or not to turn on uncertainty mode, default False',
                'uncertainty_reg': 'regularization term to balance uncertainty loss and prediction loss, default 1',
                'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
               }
    
    # -------contra loss pretraining -----
    def _calculate_perturbation_deltas(self, single_pert_list):
        """Calculates mean delta expression for single gene perturbations."""
        print_sys("Calculating mean delta expression for single perturbations...")
        delta_vectors = {}
        control_mean = self.ctrl_expression.cpu().numpy() # Mean control expression

        pert_adata = self.adata[self.adata.obs['condition'].isin(single_pert_list)]

        for pert in tqdm(single_pert_list, desc="Calculating Deltas"):
            pert_mean = np.mean(pert_adata[pert_adata.obs['condition'] == pert].X.toarray(), axis=0)
            delta_vectors[pert] = pert_mean - control_mean

        print_sys("Finished calculating deltas.")
        return delta_vectors

    def _calculate_similarity_matrix(self, delta_vectors_dict, pert_names):
        """Calculates pairwise similarity (Pearson correlation) between delta vectors."""
        print_sys("Calculating similarity matrix...")
        # Ensure order matches pert_names
        delta_matrix = np.array([delta_vectors_dict[p] for p in pert_names])
        # Handle potential NaN variance issues if a delta vector is constant zero
        valid_variance = np.std(delta_matrix, axis=1) > 1e-6
        if not np.all(valid_variance):
            print_sys(f"Warning: {np.sum(~valid_variance)} perturbations have near-zero variance in delta expression. Excluding them from similarity calculation.")
            pert_names = [p for i, p in enumerate(pert_names) if valid_variance[i]]
            delta_matrix = delta_matrix[valid_variance,:]
            if len(pert_names) < 2:
                 print_sys("Not enough valid perturbations to calculate similarity. Skipping pre-training.")
                 return None, None


        # Calculate Pearson correlation matrix
        # Using np_pearson_cor from utils if available, otherwise np.corrcoef
        try:
            from .utils import np_pearson_cor # Try importing the specific function
            sim_matrix = np_pearson_cor(delta_matrix.T, delta_matrix.T) # gene x pert -> sim between perts
        except ImportError:
            print_sys("np_pearson_cor not found in utils, using np.corrcoef.")
            sim_matrix = np.corrcoef(delta_matrix) # pert x pert -> sim between perts

        # Ensure it's symmetric and NaNs are handled (e.g., set to 0)
        sim_matrix = np.nan_to_num(sim_matrix)
        sim_df = pd.DataFrame(sim_matrix, index=pert_names, columns=pert_names)
        print_sys("Finished calculating similarity matrix.")
        return sim_df, pert_names # Return potentially filtered pert_names
    # -------contra loss pretraining -----



    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         uncertainty = False, 
                         uncertainty_reg = 1,
                         direction_lambda = 1e-1,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False,
                         **kwargs
                        ):
        """
        Initialize the model

        Parameters
        ----------
        hidden_size: int
            hidden dimension, default 64
        num_go_gnn_layers: int
            number of GNN layers for GO graph, default 1
        num_gene_gnn_layers: int
            number of GNN layers for co-expression gene graph, default 1
        decoder_hidden_size: int
            hidden dimension for gene-specific decoder, default 16
        num_similar_genes_go_graph: int
            number of maximum similar K genes in the GO graph, default 20
        num_similar_genes_co_express_graph: int
            number of maximum similar K genes in the co expression graph, default 20
        coexpress_threshold: float
            pearson correlation threshold when constructing coexpression graph, default 0.4
        uncertainty: bool
            whether or not to turn on uncertainty mode, default False
        uncertainty_reg: float
            regularization term to balance uncertainty loss and prediction loss, default 1
        direction_lambda: float
            regularization term to balance direction loss and prediction loss, default 1
        G_go: scipy.sparse.csr_matrix
            GO graph, default None
        G_go_weight: scipy.sparse.csr_matrix
            GO graph edge weights, default None
        G_coexpress: scipy.sparse.csr_matrix
            co-expression graph, default None
        G_coexpress_weight: scipy.sparse.csr_matrix
            co-expression graph edge weights, default None
        no_perturb: bool
            predict no perturbation condition, default False

        Returns
        -------
        None
        """
        
        self.config = {'hidden_size': hidden_size,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty' : uncertainty, 
                       'uncertainty_reg' : uncertainty_reg,
                       'direction_lambda' : direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.num_genes,
                       'num_perts': self.num_perts,
                       'no_perturb': no_perturb
                      }
        
        if self.wandb:
            self.wandb.config.update(self.config)
        
        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(network_type='co-express',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_co_express_graph,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions)

            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight
        
        if self.config['G_go'] is None:
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(network_type='go',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_go_graph,
                                               pert_list=self.pert_list,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions,
                                               default_pert_graph=self.default_pert_graph)

            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map = self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
            
        self.model = GEARS_Model(self.config).to(self.device)
        self.best_model = deepcopy(self.model)
    

    # ------- contrastive  pretraining --------
    def pretrain_embeddings(self, pretrain_epochs=10, pretrain_lr=1e-3, temperature=0.1, num_negatives=64, positive_threshold=0.5):
        """
        Pre-trains the gene embeddings using contrastive loss based on
        perturbation effect similarity.
        """
        print_sys("Starting contrastive pre-training of gene embeddings...")
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not initialized. Call model_initialize first.")
        if not self.pert_list:
             print_sys("Perturbation list is empty. Skipping pre-training.")
             return

        # --- 1. Data Preparation ---
        # Identify single gene perturbations present in the data
        all_single_perts_in_data = self.adata.obs[self.adata.obs['condition'].str.contains('ctrl', regex=False) & (self.adata.obs['condition'] != 'ctrl')]['condition'].unique()
        print(f"(Count: {len(all_single_perts_in_data)})")

        # Filter these to only include those also in self.pert_list (the ones with GO graph nodes/embeddings)
        single_pert_names = all_single_perts_in_data  # [p for p in all_single_perts_in_data if p in self.pert_list]

        if len(single_pert_names) < 2:
            print_sys(f"Found only {len(single_pert_names)} single gene perturbations in both data and pert_list. Need at least 2 for contrastive learning. Skipping pre-training.")
            return

        delta_vectors = self._calculate_perturbation_deltas(single_pert_names)
        sim_df, valid_pert_names = self._calculate_similarity_matrix(delta_vectors, single_pert_names)

        if sim_df is None or len(valid_pert_names) < 2 :
            print_sys("Could not calculate similarity matrix or not enough valid perturbations. Skipping pre-training.")
            return

        # Map valid perturbation gene names to their indices in the main gene embedding layer
        # gene_dict = {g:i for i,g in enumerate(self.gene_list)} # Already calculated in __init__? Check.
        # Assuming self.pert2gene map is correct from __init__

        # --- NEW STEP: Ensure we have pure gene names from valid_pert_names ---
        pure_valid_gene_names = set() # Use a set for efficiency
        starts_with_ctrl_count = 0
        ends_with_ctrl_count = 0
        other_format_count = 0 # To count names that don't fit the expected +ctrl format
        
        condition_to_gene_map = {}
        gene_to_condition_map = {}


        for name_with_ctrl in valid_pert_names:
            if name_with_ctrl.startswith('ctrl+'):
                starts_with_ctrl_count += 1
                parts = name_with_ctrl.split('+')
                gene_name = parts[1]
                #  print_sys(gene_name)
                pure_valid_gene_names.add(gene_name)
            elif name_with_ctrl.endswith('+ctrl'):
                ends_with_ctrl_count += 1
                parts = name_with_ctrl.split('+')
                gene_name = parts[0]
                #  print_sys(gene_name)
                pure_valid_gene_names.add(gene_name)
            else:
                # Handle unexpected formats if they occur
                other_format_count += 1
                pure_valid_gene_names.add(name_with_ctrl)

        # Print the summary counts
        print_sys(f"Processed {len(valid_pert_names)} items from valid_pert_names:")
        print_sys(f"  - Started with 'ctrl+': {starts_with_ctrl_count}")
        print_sys(f"  - Ended with '+ctrl': {ends_with_ctrl_count}")
        if other_format_count > 0:
            print_sys(f"  - Other format (added as-is): {other_format_count}")

        print_sys(f"Extracted {len(pure_valid_gene_names)} unique pure gene names.")
        # --- End NEW STEP ---
        
        # pert2gene is a dictionary of the index of pert and the index of gene
        pert_indices_in_gene_emb = {name: self.pert2gene[pert_idx]
                                    for pert_idx, name in enumerate(self.pert_list)
                                    if name in pure_valid_gene_names and name in self.gene_list} # Check name exists as a gene

        if len(pert_indices_in_gene_emb) < 2:
             print_sys("Not enough valid perturbations map to gene embeddings. Skipping pre-training.")
             return

        anchor_pert_names = list(pert_indices_in_gene_emb.keys())
        print_sys(f"Pre-training embeddings for {len(anchor_pert_names)} genes based on perturbation similarity.")

        # --- 2. Setup Optimizer ---
        optimizer = optim.Adam(self.model.gene_emb.parameters(), lr=pretrain_lr)
        embedding_layer = self.model.gene_emb

        # --- 3. Pre-training Loop ---
        self.model.train() # Keep model in train mode for embeddings update

        for epoch in range(pretrain_epochs):
            total_loss = 0
            processed_anchors = 0
            np.random.shuffle(anchor_pert_names) # Shuffle anchors each epoch

            for anchor_name in anchor_pert_names:
                # --- Anchor ---
                anchor_idx = torch.tensor([pert_indices_in_gene_emb[anchor_name]], dtype=torch.long).to(self.device)
                anchor_emb = embedding_layer(anchor_idx)

                # --- Positive Selection ---
                # Find perturbations similar to the anchor
                possible_key1 = anchor_name + '+ctrl'
                possible_key2 = 'ctrl+' + anchor_name
                original_anchor_condition = None

                if possible_key1 in sim_df.index:
                    original_anchor_condition = possible_key1
                elif possible_key2 in sim_df.index:
                    original_anchor_condition = possible_key2
                else:
                    print_sys(f"Warning: Neither '{possible_key1}' nor '{possible_key2}' found in sim_df index for anchor '{anchor_name}'. Skipping.")
                    continue

                # Access similarities using the found original condition name
                similarities = sim_df.loc[original_anchor_condition].drop(original_anchor_condition) # Exclude self
               
                pos_candidates = similarities[similarities > positive_threshold].index.tolist()
                # print_sys(pos_candidates)
                # Filter candidates to those that have embeddings
                # pos_candidates = [p for p in pos_candidates if p in pert_indices_in_gene_emb]
                # print_sys(pos_candidates)

                if not pos_candidates:
                    print_sys('# Skip if no suitable positive found')
                    continue # Skip if no suitable positive found

                # Choose one positive sample (e.g., randomly from candidates or the most similar)
                # positive_name = np.random.choice(pos_candidates)
                positive_name = similarities[pos_candidates].idxmax() # Take the most similar
                parts = positive_name.split('+')
                positive_name = parts[0] if parts[1] == 'ctrl' else parts[1]

                positive_idx = torch.tensor([pert_indices_in_gene_emb[positive_name]], dtype=torch.long).to(self.device)
                positive_emb = embedding_layer(positive_idx)

                # --- Negative Selection ---
                # Sample perturbations that are not the anchor or the positive
                possible_negatives = [name for name in anchor_pert_names if name != anchor_name and name != positive_name]
                if not possible_negatives:
                    print_sys('# Skip if no negative found')
                    continue # Skip if no negatives possible

                num_actual_negatives = min(num_negatives, len(possible_negatives))
                sampled_negative_names = np.random.choice(possible_negatives, num_actual_negatives, replace=False)

                negative_indices = torch.tensor([pert_indices_in_gene_emb[name] for name in sampled_negative_names], dtype=torch.long).to(self.device)
                negative_embs = embedding_layer(negative_indices) # [num_negatives, hidden_size]

                # --- Compute InfoNCE Loss ---
                # Cosine similarity: requires N x D and N x D (or 1 x D and N x D)
                sim_pos = F.cosine_similarity(anchor_emb, positive_emb, dim=1) # [1]
                sim_neg = F.cosine_similarity(anchor_emb.repeat(num_actual_negatives, 1), negative_embs, dim=1) # [num_negatives]

                logits_pos = sim_pos / temperature
                logits_neg = sim_neg / temperature

                # Combine positive and negative logits for denominator
                # Positive is the first element
                all_logits = torch.cat([logits_pos, logits_neg]) # [1 + num_negatives]

                # Calculate log-softmax. Target is index 0 (the positive sample)
                log_probs = F.log_softmax(all_logits, dim=0)

                # Loss is the negative log-probability of the positive sample
                loss = -log_probs[0]

                # --- Optimization Step ---
                if torch.isnan(loss): # Check for NaN loss
                     print_sys(f"Warning: NaN loss encountered for anchor {anchor_name}. Skipping update.")
                     continue

                optimizer.zero_grad()
                loss.backward()
                # Optional: Gradient clipping if needed
                # torch.nn.utils.clip_grad_norm_(embedding_layer.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                processed_anchors += 1

            if processed_anchors > 0:
                avg_loss = total_loss / processed_anchors
                print_sys(f"Pre-train Epoch {epoch+1}/{pretrain_epochs}, Avg Loss: {avg_loss:.4f}")
            else:
                print_sys(f"Pre-train Epoch {epoch+1}/{pretrain_epochs}, No anchors processed.")


        print_sys("Finished contrastive pre-training.")
        # Set model back to eval mode potentially, or let the main training loop handle it.
        # self.model.eval()
    # ------- contrastive  pretraining --------

    def load_pretrained(self, path):
        """
        Load pretrained model

        Parameters
        ----------
        path: str
            path to the pretrained model

        Returns
        -------
        None
        """

        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        del config['device'], config['num_genes'], config['num_perts']
        self.model_initialize(**config)
        self.config = config
        
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
    
    def save_model(self, path):
        """
        Save the model

        Parameters
        ----------
        path: str
            path to save the model

        Returns
        -------
        None

        """
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.best_model.state_dict(), os.path.join(path, 'model.pt'))
    
    def predict(self, pert_list):
        """
        Predict the transcriptome given a list of genes/gene combinations being
        perturbed

        Parameters
        ----------
        pert_list: list
            list of genes/gene combiantions to be perturbed

        Returns
        -------
        results_pred: dict
            dictionary of predicted transcriptome
        results_logvar: dict
            dictionary of uncertainty score

        """
        ## given a list of single/combo genes, return the transcriptome
        ## if uncertainty mode is on, also return uncertainty score.
        
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        for pert in pert_list:
            for i in pert:
                if i not in self.pert_list:
                    raise ValueError(i+ " is not in the perturbation graph. "
                                        "Please select from GEARS.pert_list!")
        
        if self.config['uncertainty']:
            results_logvar = {}
            
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        results_pred = {}
        results_logvar_sum = {}
        
        from torch_geometric.data import DataLoader
        for pert in pert_list:
            try:
                #If prediction is already saved, then skip inference
                results_pred['_'.join(pert)] = self.saved_pred['_'.join(pert)]
                if self.config['uncertainty']:
                    results_logvar_sum['_'.join(pert)] = self.saved_logvar_sum['_'.join(pert)]
                continue
            except:
                pass
            
            cg = create_cell_graph_dataset_for_prediction(pert, self.ctrl_adata,
                                                    self.pert_list, self.device)
            loader = DataLoader(cg, 300, shuffle = False)
            batch = next(iter(loader))
            batch.to(self.device)

            with torch.no_grad():
                if self.config['uncertainty']:
                    p, unc = self.best_model(batch)
                    results_logvar['_'.join(pert)] = np.mean(unc.detach().cpu().numpy(), axis = 0)
                    results_logvar_sum['_'.join(pert)] = np.exp(-np.mean(results_logvar['_'.join(pert)]))
                else:
                    p = self.best_model(batch)
                    
            results_pred['_'.join(pert)] = np.mean(p.detach().cpu().numpy(), axis = 0)
                
        self.saved_pred.update(results_pred)
        
        if self.config['uncertainty']:
            self.saved_logvar_sum.update(results_logvar_sum)
            return results_pred, results_logvar_sum
        else:
            return results_pred
        
    def GI_predict(self, combo, GI_genes_file='./genes_with_hi_mean.npy'):
        """
        Predict the GI scores following perturbation of a given gene combination

        Parameters
        ----------
        combo: list
            list of genes to be perturbed
        GI_genes_file: str
            path to the file containing genes with high mean expression

        Returns
        -------
        GI scores for the given combinatorial perturbation based on GEARS
        predictions

        """

        ## if uncertainty mode is on, also return uncertainty score.
        try:
            # If prediction is already saved, then skip inference
            pred = {}
            pred[combo[0]] = self.saved_pred[combo[0]]
            pred[combo[1]] = self.saved_pred[combo[1]]
            pred['_'.join(combo)] = self.saved_pred['_'.join(combo)]
        except:
            if self.config['uncertainty']:
                pred = self.predict([[combo[0]], [combo[1]], combo])[0]
            else:
                pred = self.predict([[combo[0]], [combo[1]], combo])

        mean_control = get_mean_control(self.adata).values  
        pred = {p:pred[p]-mean_control for p in pred} 

        if GI_genes_file is not None:
            # If focussing on a specific subset of genes for calculating metrics
            GI_genes_idx = get_GI_genes_idx(self.adata, GI_genes_file)       
        else:
            GI_genes_idx = np.arange(len(self.adata.var.gene_name.values))
            
        pred = {p:pred[p][GI_genes_idx] for p in pred}
        return get_GI_params(pred, combo)
    
    def plot_perturbation(self, query, save_file = None):
        """
        Plot the perturbation graph

        Parameters
        ----------
        query: str
            condition to be queried
        save_file: str
            path to save the plot

        Returns
        -------
        None

        """

        import seaborn as sns
        import matplotlib.pyplot as plt
        
        sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

        adata = self.adata
        gene2idx = self.node_map
        cond2name = dict(adata.obs[['condition', 'condition_name']].values)
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

        de_idx = [gene2idx[gene_raw2id[i]] for i in
                  adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        genes = [gene_raw2id[i] for i in
                 adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
        
        query_ = [q for q in query.split('+') if q != 'ctrl']
        pred = self.predict([query_])['_'.join(query_)][de_idx]
        ctrl_means = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()[
            de_idx].values

        pred = pred - ctrl_means
        truth = truth - ctrl_means
        
        plt.figure(figsize=[16.5,4.5])
        plt.title(query)
        plt.boxplot(truth, showfliers=False,
                    medianprops = dict(linewidth=0))    

        for i in range(pred.shape[0]):
            _ = plt.scatter(i+1, pred[i], color='red')

        plt.axhline(0, linestyle="dashed", color = 'green')

        ax = plt.gca()
        ax.xaxis.set_ticklabels(genes, rotation = 90)

        plt.ylabel("Change in Gene Expression over Control",labelpad=10)
        plt.tick_params(axis='x', which='major', pad=5)
        plt.tick_params(axis='y', which='major', pad=5)
        sns.despine()
        
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')
        plt.show()
    
    
    def train(self, epochs = 20, 
              lr = 1e-3,
              weight_decay = 5e-4
             ):
        """
        Train the model

        Parameters
        ----------
        epochs: int
            number of epochs to train
        lr: float
            learning rate
        weight_decay: float
            weight decay

        Returns
        -------
        None

        """
        
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
            
        self.model = self.model.to(self.device)
        best_model = deepcopy(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        min_val = np.inf
        print_sys('Start Training...')

        for epoch in range(epochs):
            self.model.train()

            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                optimizer.zero_grad()
                y = batch.y
                if self.config['uncertainty']:
                    pred, logvar = self.model(batch)
                    loss = uncertainty_loss_fct(pred, logvar, y, batch.pert,
                                      reg = self.config['uncertainty_reg'],
                                      ctrl = self.ctrl_expression, 
                                      dict_filter = self.dict_filter,
                                      direction_lambda = self.config['direction_lambda'])
                else:
                    pred = self.model(batch)
                    loss = loss_fct(pred, y, batch.pert,
                                  ctrl = self.ctrl_expression, 
                                  dict_filter = self.dict_filter,
                                  direction_lambda = self.config['direction_lambda'])
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()

                if self.wandb:
                    self.wandb.log({'training_loss': loss.item()})

                if step % 50 == 0:
                    log = "Epoch {} Step {} Train Loss: {:.4f}" 
                    print_sys(log.format(epoch + 1, step + 1, loss.item()))

            scheduler.step()
            # Evaluate model performance on train and val set
            train_res = evaluate(train_loader, self.model,
                                 self.config['uncertainty'], self.device)
            val_res = evaluate(val_loader, self.model,
                                 self.config['uncertainty'], self.device)
            train_metrics, _ = compute_metrics(train_res)
            val_metrics, _ = compute_metrics(val_res)

            # Print epoch performance
            log = "Epoch {}: Train Overall MSE: {:.4f} " \
                  "Validation Overall MSE: {:.4f}. "
            print_sys(log.format(epoch + 1, train_metrics['mse'], 
                             val_metrics['mse']))
            
            # Print epoch performance for DE genes
            log = "Train Top 20 DE MSE: {:.4f} " \
                  "Validation Top 20 DE MSE: {:.4f}. "
            print_sys(log.format(train_metrics['mse_de'],
                             val_metrics['mse_de']))
            
            if self.wandb:
                metrics = ['mse', 'pearson']
                for m in metrics:
                    self.wandb.log({'train_' + m: train_metrics[m],
                               'val_'+m: val_metrics[m],
                               'train_de_' + m: train_metrics[m + '_de'],
                               'val_de_'+m: val_metrics[m + '_de']})
               
            if val_metrics['mse_de'] < min_val:
                min_val = val_metrics['mse_de']
                best_model = deepcopy(self.model)
                
        print_sys("Done!")
        self.best_model = best_model

        if 'test_loader' not in self.dataloader:
            print_sys('Done! No test dataloader detected.')
            return
            
        # Model testing
        test_loader = self.dataloader['test_loader']
        print_sys("Start Testing...")
        test_res = evaluate(test_loader, self.best_model,
                            self.config['uncertainty'], self.device)
        test_metrics, test_pert_res = compute_metrics(test_res)    
        log = "Best performing model: Test Top 20 DE MSE: {:.4f}"
        print_sys(log.format(test_metrics['mse_de']))
        
        if self.wandb:
            metrics = ['mse', 'pearson']
            for m in metrics:
                self.wandb.log({'test_' + m: test_metrics[m],
                           'test_de_'+m: test_metrics[m + '_de']                     
                          })
                
        out = deeper_analysis(self.adata, test_res)
        out_non_dropout = non_dropout_analysis(self.adata, test_res)
        
        metrics = ['pearson_delta']
        metrics_non_dropout = ['frac_opposite_direction_top20_non_dropout',
                               'frac_sigma_below_1_non_dropout',
                               'mse_top20_de_non_dropout']
        
        if self.wandb:
            for m in metrics:
                self.wandb.log({'test_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

            for m in metrics_non_dropout:
                self.wandb.log({'test_' + m: np.mean([j[m] for i,j in out_non_dropout.items() if m in j])})        

        if self.split == 'simulation':
            print_sys("Start doing subgroup analysis for simulation split...")
            subgroup = self.subgroup
            subgroup_analysis = {}
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
                for m in list(list(test_pert_res.values())[0].keys()):
                    subgroup_analysis[name][m] = []

            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m, res in test_pert_res[pert].items():
                        subgroup_analysis[name][m].append(res)

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                    print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))

            ## deeper analysis
            subgroup_analysis = {}
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
                for m in metrics:
                    subgroup_analysis[name][m] = []

                for m in metrics_non_dropout:
                    subgroup_analysis[name][m] = []

            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m in metrics:
                        subgroup_analysis[name][m].append(out[pert][m])

                    for m in metrics_non_dropout:
                        subgroup_analysis[name][m].append(out_non_dropout[pert][m])

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                    print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))
        print_sys('Done!')


