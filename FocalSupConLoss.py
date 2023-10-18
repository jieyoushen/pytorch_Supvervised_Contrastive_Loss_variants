class FocalSupConLossIN(nn.Module):
    def __init__(self, temperature, sim_type, alpha, gamma, device):
        super().__init__()
        self.temperature = temperature
        self.sim_type = sim_type
        self.alpha =alpha
        self.gamma = gamma
        self.device = device

    def forward(self, features, labels):
        batch_size = features.shape[0]
        contrast_count = features.shape[1] 
        features_count = features.shape[2]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0).to(self.device) 
        
        labels = labels.to(self.device) 
        self.alpha = self.alpha.to(self.device) 
        alpha_label = self.alpha.gather(0,labels) 

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device) 
        mask2 = mask.repeat(contrast_count, contrast_count)

        mask_pos = (1 - torch.eye (batch_size* 2).to(self.device)) * mask2
        mask_noself = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(self.device)
        
        alpha_all = alpha_label.repeat(batch_size*2, 2)
        alpha_pos = alpha_all *mask_pos
        
        
        if self.sim_type == 'cosine':
            sim = (torch.matmul(contrast_feature, contrast_feature.T))/self.temperature
        elif self.sim_type == 'euclidean':
            sim = -torch.cdist(contrast_feature, contrast_feature, p = 2)/self.temperature
        elif self.sim_type == 'manhattan':
            sim = -torch.cdist(contrast_feature, contrast_feature, p = 1)/self.temperature
        
        e_sim = torch.exp(sim) * mask_noself
        e_sim_pos = mask_pos * alpha_all* e_sim    
        e_sim_batch = e_sim.sum(1, keepdim=True) ## sum of all batch
        e_sim_pos_div = e_sim_pos / e_sim_batch  ## softmax of each pair
        gam_pos = 1 * (1 - e_sim_pos_div)** self.gamma
        e_sim_pos_div = e_sim_pos_div*gam_pos
        num_pos = mask_pos.sum(1).view(-1, 1)
        e_sim_pos_div_norm = e_sim_pos_div.sum(1).view(-1, 1)/num_pos
        e_sim_pos_div_norm_log = -torch.log(e_sim_pos_div_norm)
        loss = e_sim_pos_div_norm_log.mean()
        return loss