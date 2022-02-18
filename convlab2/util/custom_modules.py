import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class Encoder(nn.Module):
    def __init__(self, s_dim, h_dim):
        super(Encoder, self).__init__()

        #self.net = nn.Sequential(nn.Linear(s_dim, h_dim),
        #                         nn.ReLU(),
        #                         nn.Linear(h_dim, h_dim),
        #                         nn.ReLU())

        self.linear1 = nn.Linear(s_dim, h_dim)
        self.linear2 = nn.Linear(h_dim, h_dim)
        self.relu = nn.ReLU()

    def forward(self, s):
        # [b, s_dim] => [b, h_dim]

        #a_weights = self.net(s)
        a_weights = self.relu(self.linear1(s))
        a_weights = self.relu(self.linear2(a_weights)) + a_weights

        return a_weights


class SkillManager(nn.Module):
    def __init__(self, s_dim, h_dim, num_skills):
        super(SkillManager, self).__init__()

        self.net = nn.Sequential(nn.Linear(s_dim, h_dim),
                                 nn.ReLU(),
                                 #nn.Linear(h_dim, h_dim),
                                 #nn.ReLU(),
                                 nn.Linear(h_dim, num_skills))

    def forward(self, s):
        # [b, s_dim] => [b, n_dim]
        a_weights = self.net(s)

        return a_weights

    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :return: [a_dim]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        a_weights = self.forward(s).squeeze()
        a_probs = torch.nn.Softmax(dim=-1)(a_weights)

        dist = Categorical(a_probs)
        a = dist.sample()

        return a

    def get_log_prob(self, s, action_mask=0):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :return: [b, 1]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]
        a_weights = self.forward(s)
        a_probs = torch.nn.Softmax(dim=-1)(a_weights)
        log_prob = torch.log(a_probs)

        return log_prob.sum(-1, keepdim=True)

    def get_entropy(self, s, action_mask=0):
        a_weights = self.forward(s)
        a_probs = torch.nn.Softmax(dim=-1)(a_weights)

        return -(a_probs * a_probs.log()).sum(-1).mean()


class SkillHead(nn.Module):
    def __init__(self, h_dim, a_dim, action_embedder=None):
        super(SkillHead, self).__init__()

        self.action_embedder = nn.Parameter(action_embedder.action_embeddings[:, :-2]) if action_embedder else None
        output_dim = self.action_embedder.size()[-2] if action_embedder else a_dim
        self.net = nn.Sequential(nn.Linear(h_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, output_dim))

    def forward(self, s):
        # [b, h_dim] => [b, a_dim]
        a_weights = self.net(s)

        if self.action_embedder is not None:
            a_weights = torch.matmul(a_weights, self.action_embedder)

        return a_weights

    def select_action(self, s, sample=True, action_mask=0):
        """
        :param s: [s_dim]
        :return: [a_dim]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights + action_mask)

        # [a_dim] => [a_dim, 2]
        a_probs = a_probs.unsqueeze(1)
        a_probs = torch.cat([1 - a_probs, a_probs], 1)
        a_probs = torch.clamp(a_probs, 1e-10, 1 - 1e-10)

        # [a_dim, 2] => [a_dim]
        a = a_probs.multinomial(1).squeeze(1) if sample else a_probs.argmax(1)

        return a

    def get_log_prob(self, s, a, action_mask=0):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :return: [b, 1]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]
        a_probs = self.get_probability(s, action_mask)

        # [b, a_dim, 2] => [b, a_dim]
        trg_a_probs = a_probs.gather(-1, a.unsqueeze(-1).long()).squeeze(-1)
        log_prob = torch.log(trg_a_probs)

        return log_prob.sum(-1, keepdim=True)

    def get_entropy(self, s, action_mask=0):

        a_probs = self.get_probability(s, action_mask)
        entropy = -(a_probs * torch.log(a_probs)).sum(-1).sum(-1)
        return entropy.mean()

    def get_probability(self, s, action_mask=0):
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights + action_mask)

        # [b, a_dim] => [b, a_dim, 2]
        a_probs = a_probs.unsqueeze(-1)
        a_probs = torch.cat([1 - a_probs, a_probs], -1)
        a_probs = torch.clamp(a_probs, 1e-10, 1 - 1e-10)

        return a_probs


class HierarchicalPolicy(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim, num_skills, action_embedder=None):
        super(HierarchicalPolicy, self).__init__()

        self.skill_manager = SkillManager(s_dim, h_dim, num_skills)

        self.encoder = Encoder(s_dim, h_dim)
        self.skills = nn.ModuleList([SkillHead(h_dim, a_dim, action_embedder) for i in range(num_skills)])

    def forward(self, s):
        # [b, s_dim] => [b, a_dim]
        pass

    def get_manager_entropy(self, s, action_mask=0):

        return self.skill_manager.get_entropy(s, action_mask)

    def select_action(self, s, sample=True, action_mask=0):
        """
        :param s: [s_dim]
        :return: [a_dim]
        """

        skill_index = self.skill_manager.select_action(s)

        if len(skill_index.size()) != 0 and skill_index.size()[0] > 1:
            a = []
            for j, index in enumerate(skill_index):
                a.append(self.skills[index].select_action(self.encoder(s[j]), action_mask=action_mask))
            a = torch.stack(a)
        else:
            a = self.skills[skill_index].select_action(self.encoder(s), action_mask=action_mask)

        return a

    def get_log_prob(self, s, a, action_mask=0):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :return: [b, 1]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]

        log_prob = self.skill_manager.get_log_prob(s)
        encoded_s = self.encoder(s)
        for skill in self.skills:
            log_prob += skill.get_log_prob(encoded_s, a, action_mask)

        return log_prob

    def get_skill_entropy(self, s, action_mask=0):

        entropy = 0
        s = self.encoder(s)
        for skill in self.skills:
            entropy += skill.get_entropy(s, action_mask)
        return entropy / len(self.skills)

    def get_kl_div(self, s, action_mask=0):

        kl_div = 0
        s = self.encoder(s)
        for i, skill in enumerate(self.skills):
            for opponent in self.skills[i+1:]:
                kl_div += kl(skill.get_probability(s, action_mask), opponent.get_probability(s, action_mask)).sum(-1)

        return kl_div.mean() / len(self.skills)


def kl(p, q):
    return (p * (p / q).log()).sum(-1)


class MultiDiscretePolicy(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super(MultiDiscretePolicy, self).__init__()

        self.net = nn.Sequential(nn.Linear(s_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, a_dim))

    def forward(self, s):
        # [b, s_dim] => [b, a_dim]
        a_weights = self.net(s)

        return a_weights

    def select_action(self, s, sample=True, action_mask=0):
        """
        :param s: [s_dim]
        :return: [a_dim]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights + action_mask)

        # [a_dim] => [a_dim, 2]
        a_probs = a_probs.unsqueeze(1)
        a_probs = torch.cat([1 - a_probs, a_probs], 1)
        a_probs = torch.clamp(a_probs, 1e-10, 1 - 1e-10)

        # [a_dim, 2] => [a_dim]
        a = a_probs.multinomial(1).squeeze(1) if sample else a_probs.argmax(1)

        return a

    def get_log_prob(self, s, a, action_mask=0):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :return: [b, 1]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights + action_mask)

        # [b, a_dim] => [b, a_dim, 2]
        a_probs = a_probs.unsqueeze(-1)
        a_probs = torch.cat([1 - a_probs, a_probs], -1)

        # [b, a_dim, 2] => [b, a_dim]
        trg_a_probs = a_probs.gather(-1, a.unsqueeze(-1).long()).squeeze(-1)
        log_prob = torch.log(trg_a_probs)

        return log_prob.sum(-1, keepdim=True)


class ActionEmbedder(nn.Module):
    def __init__(self, action_dict, domain_size, intent_size, slot_size):
        super(ActionEmbedder, self).__init__()

        domain_dict, intent_dict, slot_dict, number_dict = self.create_dicts(action_dict)

        self.embed_domain = torch.randn(len(domain_dict), domain_size)
        self.embed_intent = torch.randn(len(intent_dict), intent_size)
        self.embed_slot = torch.randn(len(slot_dict), slot_size)
        self.embed_number = torch.randn(len(number_dict), domain_size)
        self.embed_rest = torch.randn(2, domain_size + intent_size + slot_size + domain_size)

        self.action_embeddings = self.create_action_embeddings(action_dict, domain_dict, domain_size, intent_dict,
                                                          intent_size, slot_dict, slot_size, number_dict)
        self.action_embeddings.requires_grad = True

    def create_action_embeddings(self, action_dict, domain_dict, domain_size, intent_dict, intent_size, slot_dict,
                                 slot_size, number_dict):

        print(number_dict)

        action_embeddings = torch.zeros((len(action_dict) + 2, domain_size + intent_size + slot_size + domain_size))
        for action, i in action_dict.items():
            domain, intent, slot, number = action.split('-')

            domain_index = domain_dict[domain]
            intent_index = intent_dict[intent]
            slot_index = slot_dict[slot]

            domain_vec = self.embed_domain[domain_index]
            intent_vec = self.embed_intent[intent_index]
            slot_vec = self.embed_slot[slot_index]
            number_vec = self.embed_number[number_dict[number]]

            total_vec = torch.cat((domain_vec, intent_vec, slot_vec, number_vec))
            action_embeddings[i] = total_vec

        action_embeddings[len(action_dict)] = self.embed_rest[0]
        action_embeddings[len(action_dict) + 1] = self.embed_rest[1]

        return action_embeddings.permute(1, 0)

    def create_dicts(self, action_dict):
        domain_dict = {}
        intent_dict = {}
        slot_dict = {}
        number_dict = {}
        for action in action_dict:
            domain, intent, slot, number = action.split('-')
            if domain not in domain_dict:
                domain_dict[domain] = len(domain_dict)
            if intent not in intent_dict:
                intent_dict[intent] = len(intent_dict)
            if slot not in slot_dict:
                slot_dict[slot] = len(slot_dict)
            if number not in number_dict:
                number_dict[number] = len(number_dict)

        return domain_dict, intent_dict, slot_dict, number_dict

    def forward(self, actions):
        return self.embedding(actions)
