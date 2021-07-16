import numpy as np
import torch


class Forward:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.prio = self.config.REPLAY_PRIO
        self.nSteps = self.config.REPLAY_NSTEP
        self.a = 0.6
        self.e = 0.001

    def forward_multi(self, samples, weights, anns, targetAnns, social, device='cpu'):
        batches = [self.forward(samples[i], weights[i], anns[i], targetAnns[i], social[i], device=device)
                   for i in range(len(samples))]
        batches, priorities = [batch[0] for batch in batches], [batch[1] for batch in batches]
        return batches, priorities

    def forward(self, sample_of_samples, weights, ann, targetAnn, social, device='cpu'):
        Qs, returns, priorities = [], [], []

        states = torch.cat([samples[0][0] for samples in sample_of_samples], dim=0).to(device)
        atnArgs = ann(states)
        val = ann.getVal(states)

        with torch.no_grad():
            states = torch.cat([samples[-1][0] for samples in sample_of_samples], dim=0).to(device)
            atnArgs_n = ann(states)
            atnArgs_nt = targetAnn(states)
            val_nt = targetAnn.getVal(states)
            punishments = social(states)[1]

        for j, samples in enumerate(sample_of_samples):
            Qs.append(torch.zeros((self.config.N_QUANT,), dtype=torch.float32))
            returns.append(torch.zeros((self.config.N_QUANT,), dtype=torch.float32))

            s, a, r, d = samples[0]
            if d:
                if self.prio:
                    priorities.append(np.zeros(1, dtype='float').mean())
                continue

            Qs[-1] += val[j][0]
            Qs[-1] += self.calculate_A(atnArgs[0][j], a)

            for step, sample in enumerate(samples[1:]):
                s, a, r, d = sample
                returns[-1] += self.config.GAMMA ** step * r
                if d:
                    break

            if not d:
                gamma = self.config.GAMMA ** (len(samples) - 1)
                returns[-1] += gamma * val_nt[j][0]
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

    def get_priorities_from_samples(self, samples, ann, targetAnn, social, device='cpu'):
        sample_of_samples = self.rolling(samples)
        _, priorities = self.forward(sample_of_samples, None, ann, targetAnn, social, device=device)
        priorities = np.append(priorities, np.ones(1) * np.mean(priorities))
        return priorities


class ForwardSocial(Forward):
    def __init__(self, config, args):
        super(ForwardSocial, self).__init__(config, args)
        self.nSteps = self.config.REPLAY_NSTEP_LM
        self.prio = False

    def forward(self, sample_of_samples, anns, social, targetSocial, mixer, targetMixer, device='cpu'):
        Qs, returns = [], []

        for j, samples in enumerate(sample_of_samples):
            Qs.append(torch.zeros((1,), dtype=torch.float32))
            returns.append(torch.zeros((1,), dtype=torch.float32))

            ents = [ent for ent in samples[0].keys() if not samples[0][ent][4]]
            ents = ents if samples[0][ents[0]][1] == 0 else list(reversed(ents))
            global_state = samples[0][ents[0]][0]
            global_state_next = list(samples[-1].values())[0][0]

            with torch.no_grad():
                if not self.config.VANILLA_QMIX:
                    returns[-1] += self.calculate_return_barocco(samples, anns, targetSocial, ents, device)
                else:
                    returns[-1] += self.calculate_return_vanilla(samples, global_state_next, anns, social,
                                                                 targetSocial, targetMixer, ents, device)

            Qs[-1] += self.calculate_Qtot(samples[0], global_state, social, mixer, ents, device)

        return {'Qs': Qs, 'returns': returns}

    def calculate_return_barocco(self, samples, anns, social, ents, device='cpu'):
        target = []
        for ent in ents:
            val_n = torch.zeros(1)
            if ent in samples[-1].keys():
                lst = samples[-1][ent]
                if not lst[4]:
                    s = lst[0].to(device)
                    val_n = anns[lst[1]].getVal(s).mean()
                    A = anns[lst[1]](s)[0].mean(2).view(-1)
                    punish = social[lst[1]](s)[1].view(-1)
                    A_tot = punish + (1 - self.config.PUNISHMENT) * A
                    A_max = A[torch.argmax(A_tot)].mean()
                    val_n += A_max - A.mean()

            reward = sum([sample[ent][3] * self.config.GAMMA ** step if ent in sample.keys() else 0
                          for step, sample in enumerate(samples[1:])])
            target.append(val_n * self.config.GAMMA ** (len(samples) - 1) + reward)

        target = self.combine_ents(target)
        return target.view(1)

    def calculate_return_vanilla(self, samples, global_state, anns, social, targetSocial, mixer, ents,
                                 device='cpu'):
        Qtot = torch.zeros(1)
        for step, sample in enumerate(samples[1:]):
            rewards = torch.FloatTensor([sample[ent][3] if ent in sample.keys() else 0 for ent in ents])
            Qtot += self.config.GAMMA ** step * self.combine_ents(rewards)

        gamma = self.config.GAMMA ** (len(samples) - 1)
        Qtot += self.calculate_Qtot_next(samples[-1], global_state, anns, social, targetSocial, mixer, ents,
                                         device) * gamma
        return Qtot

    def calculate_Qtot(self, sample, global_state, social, mixer, ents, device='cpu'):
        agent_qs = []
        for ent in ents:
            s, annID, a, r, d = sample[ent]
            s = s.to(device)
            outLm = social[annID](s)[0]
            A = self.calculate_A(outLm, a)
            agent_qs.append(A)
        agent_qs = torch.cat(agent_qs).unsqueeze(0)
        return mixer(agent_qs.to(device), global_state.to(device)).mean().view(1).to('cpu')

    def calculate_Qtot_next(self, sample, global_state, anns, social, targetSocial, mixer, ents, device='cpu'):
        agent_qs = []
        for ent in ents:
            if ent not in sample.keys():
                agent_qs.append(torch.zeros(1))
                continue
            s, annID, a, r, d = sample[ent]
            s = s.to(device)
            outLm, _ = targetSocial[annID](s)
            _, punish = social[annID](s)
            A = anns[annID](s)[0].mean(2).view(-1)
            A_tot = punish.view(-1) + (1 - self.config.PUNISHMENT) * A
            A = self.calculate_A(outLm, torch.argmax(A_tot))
            agent_qs.append(A)
        agent_qs = torch.cat(agent_qs).unsqueeze(0)
        return mixer(agent_qs.to(device), global_state.to(device)).mean().view(1).to('cpu')

    def calculate_A(self, out, action):
        return out[:, action].view(1)

    def combine_ents(self, lst):
        return eval(self.config.LM_FUNCTION)(lst)
