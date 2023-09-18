import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from model.VIT import VIT
from model.MSU import MSU
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
EPS = 1e-20

class RLActor(nn.Module):
    def __init__(self, supports, args):
        super(RLActor, self).__init__()
        self.asu = VIT(time_series_data=args.window_len,
                       window_len=args.window_len,
                       num_stocks=args.num_assets,
                       num_features=args.in_features[0],
                       embedding_dim=args.embedding_dim,
                       layers=args.layers,
                       heads=args.heads,
                       drop_path_rate=args.drop_path_rate)
        
        if args.msu_bool:
            self.msu = MSU(in_features=args.in_features[1],
                           window_len=args.window_len,
                           hidden_dim=args.hidden_dim)
        self.args = args

    def forward(self, x_a, x_m, masks=None, deterministic=True, logger=None, y=None):
        scores,value = self.asu(x_a)
        if self.args.msu_bool:
            res,value_weight = self.msu(x_m)
        else:
            res,value_weight = (None,None)
        return self.__generator(scores, res,value_weight*value, deterministic)

    def __generator(self, scores, res, value,deterministic=None):
        weights = np.zeros((scores.shape[0], 2 * scores.shape[1]))

        winner_scores = scores
        loser_scores = scores.sign() * (1 - scores)

        scores_p = torch.softmax(scores, dim=-1)
        
        dist_a = Categorical(scores_p)

        # winners_log_p = torch.log_softmax(winner_scores, dim=-1)
        w_s, w_idx = torch.topk(winner_scores.detach(), self.args.G)

        long_ratio = torch.softmax(w_s, dim=-1)

        for i, indice in enumerate(w_idx):
            weights[i, indice.detach().cpu().numpy()] = long_ratio[i].cpu().numpy()

        l_s, l_idx = torch.topk(loser_scores.detach(), self.args.G)

        short_ratio = torch.softmax(l_s.detach(), dim=-1)
        for i, indice in enumerate(l_idx):
            weights[i, indice.detach().cpu().numpy() + scores.shape[1]] = short_ratio[i].cpu().numpy()

        if self.args.msu_bool:
            mu = res[..., 0]
            sigma = torch.log(1 + torch.exp(res[..., 1]))
            if deterministic:
                rho_class = (torch.arange(0,11)*0.1).to(self.args.device)
                
                # rho = torch.clamp(mu, 0.0, 1.0)
                m = torch.distributions.Categorical(res)
                sample_rho = m.sample()
                rho = rho_class[sample_rho]
                rho_log_p = m.log_prob(sample_rho)
            else:
                # bayes sample 
                m = Normal(mu, sigma)
                sample_rho = m.sample()
                rho = torch.clamp(sample_rho, 0.0, 1.0)
                rho_log_p = m.log_prob(sample_rho)
        else:
            rho = torch.ones((weights.shape[0])).to(self.args.device) * 0.5
            rho_log_p = None
        return weights, rho, scores_p, rho_log_p, dist_a.entropy()+m.entropy(),value


class RLAgent():
    def __init__(self, env, actor, args, logger=None):
        self.actor = actor
        self.env = env
        self.args = args
        self.logger = logger

        self.total_steps = 0
        self.optimizer = torch.optim.Adam(self.actor.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.weight_decay)
        self.minibatch = 4
    def train_episode(self):
        self.__set_train()
        states, masks = self.env.reset()

        steps = 0
        

        steps_log_p_rho = []
        steps_reward_total = []
        steps_asu_grad = []

        rho_records = []

        # agent_wealth = np.ones((batch_size, 1), dtype=np.float32)

        while True:
            print("steps:", steps)
            steps += 1
            x_a = torch.from_numpy(states[0]).to(self.args.device).detach()
            masks = torch.from_numpy(masks).to(self.args.device).detach()
            if self.args.msu_bool:
                x_m = torch.from_numpy(states[1]).to(self.args.device).detach()
            else:
                x_m = None
            weights, rho, scores_p, log_p_rho,entropy, value \
                = self.actor(x_a, x_m, masks, deterministic=True)

            ror = torch.from_numpy(self.env.ror).to(self.args.device)
            normed_ror = (ror - torch.mean(ror, dim=-1, keepdim=True)) / \
                         torch.std(ror, dim=-1, keepdim=True)

            next_states, rewards, rho_labels, masks, done, info = \
                self.env.step(weights, rho.detach().cpu().numpy())

            # steps_log_p_rho.append(log_p_rho)
            # steps_reward_total.append(rewards.total - info)
            rewards_step = rewards.total - info['market_avg_return']

            # asu_grad = torch.sum(normed_ror * scores_p, dim=-1)
            # steps_asu_grad.append(torch.log(asu_grad))

            agent_wealth = info["agent_wealth"]
            

            # rho_records.append(np.mean(rho.detach().cpu().numpy()))


            
                # if done:
            # if self.args.msu_bool:
            #     steps_log_p_rho = torch.stack(steps_log_p_rho, dim=-1)

            # steps_reward_total = np.array(steps_reward_total).transpose((1, 0))

            # rewards_total = torch.from_numpy(steps_reward_total).to(self.args.device)
            mdd = self.cal_MDD(agent_wealth)

            rewards_mdd = - 2 * torch.from_numpy(mdd - 0.5).to(self.args.device)
            rewards_step = torch.from_numpy(rewards_step).to(self.args.device)
            # rewards_total = (rewards_total - torch.mean(rewards_total, dim=-1, keepdim=True)) \
            #                 / torch.std(rewards_total, dim=-1, keepdim=True)
            rewards_step = (rewards_step - torch.mean(rewards_step, dim=-1, keepdim=True)) \
                            / torch.std(rewards_step, dim=-1, keepdim=True)
            old_action_log_prob = torch.cat([torch.log(scores_p),torch.unsqueeze(log_p_rho,-1)],dim=-1).detach()
            with torch.no_grad():
                next_x_a = torch.from_numpy(states[0]).to(self.args.device)
                next_masks = torch.from_numpy(masks).to(self.args.device)
                if self.args.msu_bool:
                    next_x_m = torch.from_numpy(states[1]).to(self.args.device)
                else:
                    next_x_m = None
                _, _, _, _,_, next_value \
                    = self.actor(next_x_a, next_x_m, next_masks, deterministic=True)
                target_v = rewards_step + self.args.gamma * next_value
            # gradient_asu = torch.stack(steps_asu_grad, dim=1)
            # gradient_asu = torch.nan_to_num(gradient_asu)
            advantage = (target_v - value).detach()
            for i in range(self.args.ppo_epoch):
                for index in BatchSampler(SubsetRandomSampler(range(self.args.batch_size)), self.minibatch,False):
                    weights, rho, scores_p, log_p_rho, entropy,value \
                    = self.actor(x_a[index], x_m[index], masks[index], deterministic=True)
                    action_log_prob = torch.cat([torch.log(scores_p),torch.unsqueeze(log_p_rho,-1)],dim=-1)
                    ratio = torch.exp(torch.sum(action_log_prob,-1) - torch.sum(old_action_log_prob[index],-1))
                    L1 = ratio * advantage[index]
                    L2 = torch.clamp(ratio, 1-self.args.clip_param, 1+self.args.clip_param) * advantage[index]
                    action_loss = -torch.min(L1, L2).mean()
                    value_loss = F.smooth_l1_loss(value, target_v[index])
                    loss = action_loss + 0.5*value_loss - 0.01* torch.mean(entropy,-1)
                    # if self.args.msu_bool:
                    #     gradient_rho = (rewards_mdd * steps_log_p_rho)
                    #     loss = - (self.args.gamma * gradient_rho + gradient_asu)
                    # else:
                    #     loss = - (gradient_asu)
                    assert not torch.isnan(loss)
                    
                    loss = loss.contiguous()
                    loss.backward()
                    grad_norm, grad_norm_clip = self.clip_grad_norms(self.optimizer.param_groups, self.args.max_grad_norm)
                    self.optimizer.step()
            if done:
                break
            states = next_states

        rtns = (agent_wealth[:, -1] / agent_wealth[:, 0]).mean()
        avg_rho = np.mean(rho_records)
        avg_mdd = mdd.mean()
        return rtns, avg_rho, avg_mdd,loss.cpu().item()

    def evaluation(self, logger=None):
        self.__set_eval()
        states, masks = self.env.reset()

        steps = 0
        batch_size = states[0].shape[0]

        # agent_wealth = np.ones((batch_size, 1), dtype=np.float32)
        rho_record = []
        while True:
            steps += 1
            x_a = torch.from_numpy(states[0]).to(self.args.device)
            masks = torch.from_numpy(masks).to(self.args.device)
            if self.args.msu_bool:
                x_m = torch.from_numpy(states[1]).to(self.args.device)
            else:
                x_m = None

            weights, rho, _, _,_,_ \
                = self.actor(x_a, x_m, masks, deterministic=True)
            next_states, rewards, _, masks, done, info = self.env.step(weights, rho.detach().cpu().numpy())
            agent_wealth = info["agent_wealth"]
            
            states = next_states

            if done:
                break

        return agent_wealth

    def test(self, logger=None):
            self.__set_test()
            states, masks = self.env.reset()

            steps = 0
            batch_size = states[0].shape[0]
            #print("In Ag 211:", batch_size)
            agent_wealth = np.ones((batch_size, 1), dtype=np.float32)
            #print("In Ag 213:", agent_wealth.shape)
            rho_record = []
            while True:
                steps += 1
                print("************** STEP:",steps)
                if steps==633:
                    print("123")
                x_a = torch.from_numpy(states[0]).to(self.args.device)
                masks = torch.from_numpy(masks).to(self.args.device)
                if self.args.msu_bool:
                    x_m = torch.from_numpy(states[1]).to(self.args.device)
                else:
                    x_m = None

                #print("X_A:", x_a.shape, "X_M:",x_m.shape, "Masks:",masks.shape)
                weights, rho, _, _,_,_ \
                    = self.actor(x_a, x_m, masks, deterministic=True)
                #print("Weights:",weights.shape, "Rho:",rho.shape)
                next_states, rewards, _, masks, done, info = self.env.step(weights, rho.detach().cpu().numpy())
                #print("Ns:",len(next_states))
                #print("Rewards:",rewards)
                #print("done:",done)
                #print("Masks:",masks.shape)
                
                agent_wealth = info["agent_wealth"]
                states = next_states

                if done:
                    break
            
            return agent_wealth
    def clip_grad_norms(self, param_groups, max_norm=math.inf):
        """
        Clips the norms for all param groups to max_norm
        :param param_groups:
        :param max_norm:
        :return: gradient norms before clipping
        """
        grad_norms = [
            torch.nn.utils.clip_grad_norm_(
                group['params'],
                max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
                norm_type=2
            )
            for group in param_groups
        ]
        grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
        return grad_norms, grad_norms_clipped

    def __set_train(self):
        self.actor.train()
        self.env.set_train()

    def __set_eval(self):
        self.actor.eval()
        self.env.set_eval()

    def __set_test(self):
        self.actor.eval()
        self.env.set_test()

    def cal_MDD(self, agent_wealth):
        drawdown = (np.maximum.accumulate(agent_wealth, axis=-1) - agent_wealth) / \
                   np.maximum.accumulate(agent_wealth, axis=-1)
        MDD = np.max(drawdown, axis=-1)
        return MDD[..., None].astype(np.float32)

    def cal_CR(self, agent_wealth):
        pr = np.mean(agent_wealth[:, 1:] / agent_wealth[:, :-1] - 1, axis=-1, keepdims=True)
        mdd = self.cal_MDD(agent_wealth)
        softplus_mdd = np.log(1 + np.exp(mdd))
        CR = pr / softplus_mdd
        return CR
