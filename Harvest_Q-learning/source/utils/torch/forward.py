import numpy as np
import torch


class Forward:
    def __init__(self, config, args):
        self.args, self.config = args, config
        self.prio = self.config.REPLAY_PRIO
        self.nSteps = self.config.REPLAY_NSTEP
        self.nQuant = self.config.N_QUANT
        self.a = 0.6
        self.e = 0.001

    def forward_multi(self, samples, weights, anns, targetAnns, lawmaker, device='cpu'):
        batches = [self.forward(samples[i], weights[i], anns[i], targetAnns[i], lawmaker[i], device=device)
                   for i in range(len(samples))]
        batches, priorities = [batch[0] for batch in batches], [batch[1] for batch in batches]
        return batches, priorities

    def forward(self, sample_of_samples, weights, ann, targetAnn, lawmaker, device='cpu'):
        Qs, returns, priorities = [], [], []

        stim = torch.cat([samples[0][0] for samples in sample_of_samples], dim=0)
        out, val = ann(stim.to(device))

        stim = torch.cat([samples[-1][0] for samples in sample_of_samples], dim=0)
        with torch.no_grad():
            atnArgs_n, val_n = ann(stim.to(device))
            atnArgs_nt, val_nt = targetAnn(stim.to(device))
            punishments = lawmaker(stim.to(device))[1]

        for j, samples in enumerate(sample_of_samples):
            Qs.append(torch.zeros((self.nQuant,), dtype=torch.float32))
            returns.append(torch.zeros((self.nQuant,), dtype=torch.float32))
            if sum([sample[3] for sample in samples]):
                if self.prio:
                    priorities.append(torch.zeros(1))
                continue

            s, a, r, d = samples[0]

            Qs[-1] += val[j][0]
            Qs[-1] += self.calculate_A(out[0][j], a)

            if d:
                if self.prio:
                    priorities.append(self.calculate_priority(Qs[-1].detach().numpy()))
                continue

            for step, sample in enumerate(samples[1:]):
                s, a, r, d = sample
                returns[-1] += self.config.GAMMA ** step * r
                if d:
                    break

            if not d:
                gamma = self.config.GAMMA ** (len(samples) - 1)
                returns[-1] += gamma * val_nt[j][0].detach()
                returns[-1] += self.calculate_return(atnArgs_n[0][j], punishments[j], atnArgs_nt[0][j], gamma)

            if self.prio:
                priorities.append(self.calculate_priority(Qs[-1].detach().numpy() - returns[-1].numpy()))
        return {'Qs': Qs, 'returns': returns, 'weights': weights}, priorities

    def calculate_A(self, out, action):
        return out[action.view(1)].view(-1,) - out.mean(0).view(-1)

    def calculate_return(self, out_n, punishments, out_nt, gamma):
        A_tot = out_n.mean(1).view(-1,) * (1 - self.config.PUNISHMENT) + punishments.view(-1,)
        return gamma * (out_nt[torch.argmax(A_tot), :].view(-1) - out_nt.mean(0).view(-1)).detach()

    def calculate_priority(self, td):
        return (np.abs(td).mean() + self.e) ** self.a

    def rolling(self, samples):
        sample_of_samples = []
        for i in range(len(samples) - 1):
            sample_of_samples.append(samples[i: min(i + 1 + self.nSteps, len(samples))])

        return sample_of_samples

    def get_priorities_from_samples(self, samples, ann, targetAnn, lawmaker, device='cpu'):
        sample_of_samples = self.rolling(samples)
        _, priorities = self.forward(sample_of_samples, None, ann, targetAnn, lawmaker, device=device)
        priorities = np.append(priorities, np.zeros(1))
        return priorities


class ForwardSocial(Forward):
    def __init__(self, config, args):
        super(ForwardSocial, self).__init__(config, args)
        self.nSteps = self.config.REPLAY_NSTEP_LM
        self.prio = self.config.REPLAY_PRIO_LM
        self.nQuant = self.config.N_QUANT_LM

    def forward(self, sample_of_samples, weights, anns, social, targetSocial, mixer, targetMixer, device='cpu'):
        Qs, returns, priorities = [], [], []

        for j in range(len(sample_of_samples)):
            global_state = sample_of_samples[j][0]['global_state']
            global_state_next = sample_of_samples[j][-1]['global_state']
            samples = [sample['agents'] for sample in sample_of_samples[j]]
            ents = list(samples[0].keys())

            Qs.append(torch.zeros((self.nQuant,), dtype=torch.float32))
            returns.append(torch.zeros((self.nQuant,), dtype=torch.float32))
            if sum([sample[list(sample.keys())[0]][4] for sample in samples]):
                if self.prio:
                    priorities.append(torch.zeros(1))
                continue

            with torch.no_grad():
                if not self.config.VANILLA_QMIX:
                    returns[-1] += self.calculate_return_barocco(samples, anns, targetSocial, ents, device)
                else:
                    returns[-1] += self.calculate_return_vanilla(samples, global_state_next, anns, social,
                                                                 targetSocial, targetMixer, ents, device)

            Qs[-1] += self.calculate_Qtot(samples[0], global_state, social, mixer, ents, device)
            if self.prio:
                priorities.append(self.calculate_priority(Qs[-1].detach().to('cpu').numpy() - returns[-1].to('cpu').numpy()))

        return {'Qs': Qs, 'returns': returns, 'weights': weights}, priorities

    def calculate_return_barocco(self, samples, anns, lawmaker, ents, device='cpu'):
        target = []
        for ent in ents:
            val_n = torch.zeros(self.nQuant)
            if ent in samples[-1].keys():
                lst = samples[-1][ent]
                if not lst[4]:
                    A, val_n = anns[lst[1]](lst[0].to(device))
                    A = A[0]
                    punish = lawmaker[lst[1]](lst[0].to(device))[1]
                    A_tot = punish + (1 - self.config.PUNISHMENT) * A.mean(2)
                    A_max = A[:, torch.argmax(A_tot.view(-1))].unsqueeze(1)
                    val_n += A_max - A.mean(1, keepdim=True)

            reward = sum([sample[ent][3] * self.config.GAMMA ** step if ent in sample.keys() else 0
                          for step, sample in enumerate(samples[1:])])
            val_n = val_n.view(-1)
            target.append(val_n * self.config.GAMMA ** (len(samples) - 1) + reward)

        target = self.combine_ents(target)
        if self.nQuant == 1:
            target = target.mean()
        return target.view(-1)

    def calculate_return_vanilla(self, samples, global_state, anns, lawmaker, targetLawmaker, mixer, ents, device='cpu'):
        Qtot = torch.zeros(self.nQuant)
        for step, sample in enumerate(samples[1:]):
            rewards = [sample[ent][3] if ent in sample.keys() else 0 for ent in ents]
            Qtot += self.config.GAMMA ** step * eval(self.config.LM_FUNCTION)(rewards)

        gamma = self.config.GAMMA ** (len(samples) - 1)
        Qtot += self.calculate_Qtot_next(samples[-1], global_state, anns, lawmaker, targetLawmaker, mixer, ents, device) * gamma
        return Qtot

    def calculate_Qtot(self, sample, global_state, lawmaker, mixer, ents, device='cpu'):
        agent_qs = []
        for ent in ents:
            s, annID, a, r, d = sample[ent]
            outLm = lawmaker[annID](s.to(device))[0]
            A = self.calculate_A(outLm, a)
            agent_qs.append(A)
        agent_qs = torch.cat(agent_qs).unsqueeze(0)
        return mixer(agent_qs, global_state).view(-1)

    def calculate_Qtot_next(self, sample, global_state, anns, lawmaker, targetLawmaker, mixer, ents, device='cpu'):
        agent_qs = []
        for ent in ents:
            s, annID, a, r, d = sample[ent]
            outLm, _ = targetLawmaker[annID](s.to(device))
            _, punish = lawmaker[annID](s.to(device))
            A = anns[annID](s.to(device))[0][0].mean(2).view(-1)
            A_tot = punish.view(-1) + (1 - self.config.PUNISHMENT) * A
            A = self.calculate_A(outLm, torch.argmax(A_tot))
            agent_qs.append(A)
        agent_qs = torch.cat(agent_qs).unsqueeze(0)
        return mixer(agent_qs, global_state).view(-1)

    def calculate_A(self, out, action):
        return out[:, action].mean().view(1)

    def get_priorities_from_samples(self, samples, anns, lawmaker, mixer, device='cpu'):
        sample_of_samples = self.rolling(samples)
        _, priorities = self.forward(sample_of_samples, None, anns, lawmaker, lawmaker, mixer, mixer, device=device)
        priorities = np.append(priorities, np.zeros(1))
        return priorities

    def combine_ents(self, lst):
        if self.config.LM_FUNCTION == 'sum':
            return sum(lst)
        elif self.config.LM_FUNCTION == 'min':
            idx = torch.argmin(torch.stack([t.mean() for t in lst]))
            return lst[idx]
