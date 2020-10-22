---
layout: post
title:  "Tackling the game Kalah using reinforcement learning - Part 1"
date:   2020-10-02 18:00:00 +0200
categories: ["Machine Learning", "Reinforcement Learning"]
---

* TOC
{:toc}

In this article series we are going to talk about reinforcement learning (RL) [1], an exciting part of the whole machine learning area and one of the three major parts, besides supervised (see [Predicting the outcome of a League of Legends match]({% post_url 2020-07-11-machine_learning_lol_10min_match_predictions %}) for an example) and unsupervised learning. The idea behind RL is to train a model, usually called an agent, to take actions in an environment so that the cumulative reward over time (must not necessarily mean real time) is maximized. In contrast to supervised learning, in RL the agent is not fed with labels and is not told what is the "correct" move, but the idea is, that the agent learns by itself in the given environment solely by providing an observation and the gained/lost reward after a taken action in the environment.

Here we will use this approach to tackle the game Kalah [2]. To mix things up a little, this time we are going to use PyTorch [3] as our library of choice.

We will show that it is possible to train a RL agent to play better than established, hard-coded approaches to Kalah, if certain parameters are well chosen. We will also give an outlook on improvements to the algorithms and what different approaches we could use.

# Introductory remarks

In this article we will first introduce the game Kalah, followed by implementations of classical agents for the game, which serve as a baseline for comparing our machine learning models to it.

# Kalah

Kalah [2] is a two-player game in the Mancala family invented by William Julius Champion, Jr. in 1940.

The game is played on a board and with a number of "seeds". The board has a certain number of small pits, called houses, on each side (usually 6, but we will also use 4) and a big pit, called the end zone, at each end. The objective  of the game is to capture more seeds than your opponent.

{:refdef: style="text-align: center;"}
![](/assets/img/kalah_board.jpg)
{: refdef}
{:refdef: style="text-align: center;"}
*Figure 1: A Kalah board in the game start configuration.*
{: refdef}

The rules are the following:

1) You start with 4 or 6 (or whatever you agree on) number of seeds in every of the player pits.

2) The players take turns sowing their seeds. The current player takes all the seeds from one of his pits and places them, one by one counter-clockwise into each of the following pits, including his own end zone pit, but not in the opponents end zone pit.

3) If the last sown seed lands in an empty house owned by the current player, and if the opposite house contains seeds, all the seeds in the pit were he placed the last seed and the seeds in the opposite pit belongs to the player and shall be placed into his end zone.

4) If the last sown seed lands in the player's end zone, the player can take an additional move.

5) When a player does not have any more seeds in his pits, the game ends and the opposing player can take all its remaining seeds and place it in its end zone.

6) The player with the most seeds in their end zone wins.

For many variants of the game, it was shown that the first player has a strong advantage when both are playing a perfect game. However, for the (pits, seeds) = (6,6) variant, this is not yet that clear how big the advantage is. There are also additional rules which can mitigate that advantage, but we will not go into detail and if you are interested in that, feel free to consult Wikipedia.

In this article we are going to play with the (4,4), (6,4) and (6,6) variants.

# Classic agents

Before we start talking about reinforcement learning approaches to playing Kalah, we will first present a few classic agents which will serve as our baseline in comparison.

## Random agent

This random agent, as the name suggests, will randomly choose a move out of the valid moves. This is the simplest approach we can take on implementing a Kalah playing agent and it can be implemented essentially with just one line of Python code.

## Max score agent

The idea behind this agent is, that it will always take the move with which gives him the highest score. This can either be a move which will let him sow a seed into his own end zone, or, ideally, it will be a move were it can steel the opponents seeds by hitting an empty pit.

## Max score and repeat agent

The base strategy for this agent is the same as the max score agent, the difference is, that it will prefer a move were it will hit his own end zone with his last seed, meaning that it can get another move. This is implemented in such a way to exploit the possibility of having more than one additional move if the board permits that. This can easily be implemented by always taking a look at the possible moves from the left of the board going right and picking the first where a repeating play is possible.

## Minimax agent

The minimax algorithm [4] is a very common decision rule in game theory, statistics and many other fields. One tries to minimize the possible loss for a worst case (maximum loss) scenario.

The pseudo code for the algorithm (take from Wikipedia) is given by:

```
function minimax(node, depth, maximizingPlayer) is
    if depth = 0 or node is a terminal node then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
        for each child of node do
            value := max(value, minimax(child, depth − 1, FALSE))
        return value
    else (* minimizing player *)
        value := +∞
        for each child of node do
            value := min(value, minimax(child, depth − 1, TRUE))
        return value
```

If not otherwise specified, we will use a minimax depth of $D_{max}=4$ and the agent will use alpha-beta pruning to speed up the calculations.

# Reinforcement learning agents

{:refdef: style="text-align: center;"}
![](/assets/img/Reinforcement_learning_diagram.svg.png)
{: refdef}
{:refdef: style="text-align: center;"}
*Figure 2: Reinforcement learning. Courtesy of Wikipedia.*
{: refdef}

## REINFORCE algorithm

There are many different approaches to reinforcement learning. In our case, we will take, in my opinion, the most straightforward and easy to gasp approach: Policy gradients.

In the policy gradient method, we are directly trying to find the best policy (something which tells us what action to choose in each step of the problem).

The algorithm we are going to apply was described in [5] and a good explanation and implementation can be found in [6].

Additionally, a very nice overview over different algorithms, including REINFORE is presented at: [https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#reinforce](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#reinforce)

Here we are going to briefly outline the idea behind the algorithm:

1) Initialize the network with random weights.

2) Play an episode and save its $(s, a, r, s')$ transition.

3) For every step $t=1,2,...,T$: Calculate the discounted reward/return $Q_t=\sum^\infty_{k=0}\gamma^kR_{t+k+1}$, where $\gamma$ is the discount factor. $\gamma = 1$ means no discount, all time steps count the same, and $\gamma < 1$ means higher discounts.

4) Calculate the loss function
$L=-\sum_tQ_t\ln(\pi(s_t,a_t))$

5) Calculate the gradients, use stochastic gradient decent and update the weights of the model, minimizing the loss (therefore, we need the minus sign in step 4 in front of the sum).

6) Repeat from step 2 until problem is considered solved.

$s$ is a state, $s'$ is the new state after taking action $a$ and $r$ is the reward obtained at a specific time step.

An example implementation in PyTorch can be found at [here](https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py), where REINFORCE is used to solve the cart pole example.

## Actor-critic algorithm

{:refdef: style="text-align: center;"}
![](/assets/img/actor_critic.png)
{: refdef}
{:refdef: style="text-align: center;"}
*Figure 3: Sketch of the actor-critic model structure.*
{: refdef}

An example implementation in PyTorch can be found [here](https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py).

# Training of the RL agents

Training the RL agents turned out to be a challenge. We were not able to get the REINFORCE agent to converge, even after playing 50000 episodes against different types of classical agents. Also changing $\gamma$, learning rate, rewards or trying with different seeds did not lead to a converging REINFORCE algorithm and the win rate stayed always below 40 %. As a positive aspect: It could learn the invalid moves reasonably well.

With the actor-critic agent it was easier to find parameters for which the algorithm converged, at least on $(4,4)$ boards. There AC worked very well, as we will show later on in the comparison sector, and also changing learning rates, or $\gamma$ only changed the speed of which the algorithm converged. The rewards we used in the end are the following:

- Get number seeds placed into own house as rewards minus 0.2 (to make it less favorable to gain no points)
- For a win get +10
- For a loss get -10
- For an invalid move get -5 and the game is over

It turned out, that it was hard to train against the random agent. It worked best against the max score and max score and repeat agents. In the end we settled with the Max score and repeat agent for our training.

When training on larger boards $(6,4)$ and $(6,6)$ we could not get the AC algorithm to converge and it was usually stuck at a low win percentage, even after tuning the parameters or after trying with various random seeds.

# Comparison

For the comparison we let every agent play $N=1000$ games against every other agent, including itself, with the exception of the AC agent, as currently it can only play as player 1 (updating the environment, so that it is possible to play as player 2 is part of the planed improvements).

In Table 1 we compare the classic agents against AC on a $(4,4)$ board.

<table>
  <caption>Table 1: Comparison of classic agents on a $(4,4)$ board. Shown is the average win percentage of player 1 (rows) vs. player 2 (columns) after playing $N=1000$ games.</caption>
  {% for row in site.data.ml.comp_4_4 %}
    {% if forloop.first %}
    <tr>
      {% for pair in row %}
        <th>{{ pair[0] }}</th>
      {% endfor %}
    </tr>
    {% endif %}

    {% tablerow pair in row %}
      {{ pair[1] }}
    {% endtablerow %}
  {% endfor %}
</table>

<table>
    <caption>Figure 2: Comparison of classic agents on a $(6,4)$ board. Shown is the average win percentage of player 1 (rows) vs. player 2 (columns) after playing $N=1000$ games.</caption>
  {% for row in site.data.ml.comp_6_4 %}
    {% if forloop.first %}
    <tr>
      {% for pair in row %}
        <th>{{ pair[0] }}</th>
      {% endfor %}
    </tr>
    {% endif %}

    {% tablerow pair in row %}
      {{ pair[1] }}
    {% endtablerow %}
  {% endfor %}
</table>

<table>
  <caption>Figure 3: Comparison of classic agents on a $(6,6)$ board. Shown is the average win percentage of player 1 (rows) vs. player 2 (columns) after playing $N=1000$ games.</caption>
  {% for row in site.data.ml.comp_6_6 %}
    {% if forloop.first %}
    <tr>
      {% for pair in row %}
        <th>{{ pair[0] }}</th>
      {% endfor %}
    </tr>
    {% endif %}

    {% tablerow pair in row %}
      {{ pair[1] }}
    {% endtablerow %}
  {% endfor %}
</table>

<table>
  <caption>Figure 4: Comparison of classic agents on a $(6,6)$ board were minimax agent uses a maximum depth of $D_{max}=6$. Shown is the average win percentage of player 1 (rows) vs. player 2 (columns) after playing $N=1000$ games.</caption>
  {% for row in site.data.ml.comp_6_6_mini_depth6 %}
    {% if forloop.first %}
    <tr>
      {% for pair in row %}
        <th>{{ pair[0] }}</th>
      {% endfor %}
    </tr>
    {% endif %}

    {% tablerow pair in row %}
      {{ pair[1] }}
    {% endtablerow %}
  {% endfor %}
</table>

# Summary and discussion

# Outlook

The next step will be implementing improved versions of REINFORCE. Especially we want to batch together episodes in the update step which should improve signal to noise and reduce the variance, i.e., should allow for a much more stable model over training time and hopefully will lead to an improved performance and give us a stable parameter space. In addition we will look into of the Actor-Critic method and its various improvements that have been proposed to it, especially we will see how Advantage Actor Critic (A2C) and Asynchronous Advantage Actor Critic (A3C) are implemented and how they perform in comparison to our classic agents and to the REINFORCE algorithm.

From the implementation point of view improvements to the environment and to the Kalah board will be made to allow the RL agent to play as a second player. An idea will be to rotate the observation in case of player two so that it appears like the agent would sit in front of the board on its own side. This will be a step towards the exiting topic of playing the RL agent against itself and possible even train it like that.

Of course the highest priority will have to cleanup the code so that we can publish it on Github.

# References

[1] [https://en.wikipedia.org/wiki/Reinforcement_learning](https://en.wikipedia.org/wiki/Reinforcement_learning)

[2] [https://en.wikipedia.org/wiki/Kalah](https://en.wikipedia.org/wiki/Kalah)

[3] [https://pytorch.org/](https://pytorch.org/)

[4] [https://en.wikipedia.org/wiki/Minimax](https://en.wikipedia.org/wiki/Minimax)

[5] Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Reinforcement Learning. Springer, Boston, MA, 1992. 5-32.

[6] Lapan, Maxim. "Deep Reinforcement Learning Hands-On", Second Edition, Packt, Birmingham, UK, 2020, 286-308.