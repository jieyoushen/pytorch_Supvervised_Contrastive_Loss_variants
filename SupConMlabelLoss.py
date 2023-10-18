 
class SupConMlabelLoss(nn.Module):
    def __init__(self, temperature, device):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = 0.07
        self.sim_type = 'cosine'
        self.device = device
    

    @staticmethod    
    def get_label_mask(labels):
        labels = labels.detach().cpu().numpy()
        m = np.dot(labels, labels.T)
        m = np.where(m>1,1,m) 
        return torch.tensor(m)
        
    def forward(self, features, labels, label2):  #label2 are used to find equal groups
        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0).to(self.device) # (AD1, CN1, AD2, CN2 ...)
         
        mask = self.get_label_mask(labels).to(self.device)
        mask = mask.repeat(contrast_count, contrast_count)
        mask2 = torch.eq(label2, label2.T).long().to(self.device)
        mask2 = mask2.repeat(contrast_count, contrast_count)
        
        mask = mask+mask2
        mask = torch.where(mask>1,1,mask)
        mask_same_class = (1 - torch.eye(batch_size* 2).to(self.device) ) * mask

        if self.sim_type == 'cosine':
            sim = (torch.matmul(contrast_feature, contrast_feature.T))/self.temperature
        elif self.sim_type == 'euclidean':
            sim = torch.cdist(contrast_feature, contrast_feature, p = 2)/self.temperature
        elif self.sim_type == 'manhattan':
            sim = torch.cdist(contrast_feature, contrast_feature, p = 1)/self.temperature

        logits_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(self.device)
        exp_logits = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask_same_class * log_prob).sum(1) / mask_same_class.sum(1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

        loss = loss.mean()
        return loss