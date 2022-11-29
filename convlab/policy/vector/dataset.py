import torch.utils.data as data


class ActDataset(data.Dataset):
    def __init__(self, s_s, a_s, m_s):
        self.s_s = s_s
        self.m_s = m_s
        self.a_s = a_s
        self.num_total = len(s_s)
    
    def __getitem__(self, index):
        s = self.s_s[index]
        m = self.m_s[index]
        a = self.a_s[index]
        return s, a, m
    
    def __len__(self):
        return self.num_total


class ActDatasetKG(data.Dataset):
    def __init__(self, action_batch, a_masks, current_domain_mask_batch, non_current_domain_mask_batch):
        self.action_batch = action_batch
        self.a_masks = a_masks
        self.current_domain_mask_batch = current_domain_mask_batch
        self.non_current_domain_mask_batch = non_current_domain_mask_batch
        self.num_total = len(action_batch)

    def __getitem__(self, index):
        action = self.action_batch[index]
        action_mask = self.a_masks[index]
        current_domain_mask = self.current_domain_mask_batch[index]
        non_current_domain_mask = self.non_current_domain_mask_batch[index]

        return action, action_mask, current_domain_mask, non_current_domain_mask, index

    def __len__(self):
        return self.num_total


class ActStateDataset(data.Dataset):
    def __init__(self, s_s, a_s, next_s):
        self.s_s = s_s
        self.a_s = a_s
        self.next_s = next_s
        self.num_total = len(s_s)
    
    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        next_s = self.next_s[index]
        return s, a, next_s
    
    def __len__(self):
        return self.num_total
