# Episode 1: Flow matching

Flow matching is the training method behind image and video models such as Stable Diffusion 3, FLUX and Wan. This folder trains it on a 2D spiral.

Video: [Flow Matching from intuition to code](VIDEO_LINK)
Paper: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman et al.)

## The whole method

```python
x1 = make_spiral(64)                    # data
x0 = torch.randn_like(x1)               # noise
t  = torch.rand(len(x1), 1)             # time in [0, 1]
xt = t * x1 + (1 - t) * x0              # point on the straight line
v  = model(xt, t)                       # predicted arrow
loss = torch.mean((v - (x1 - x0)) ** 2)
```

To generate, start from noise and follow the predicted arrows with Euler steps.

## Files

- `flow_matching.py`: the code shown in the video (data, network, loss, training loop, sampler)
- `notebook.ipynb`: the same method with plots and the animation. Some names differ from the script (`generate_spiral`, `euler_sampler`), the loss is written inside the training loop, and the middle layer of the network is wider.
- `requirements.txt`: PyTorch and Matplotlib

## Run it

```bash
pip install -r requirements.txt
python flow_matching.py
```

It prints:

```
training for 2000 epochs ...
final epoch loss: 2.1849
mean of last 100 epochs: 2.2287
generated 1000 points with 50 Euler steps (plots are in notebook.ipynb)
```

Training is 2,000 epochs (32,000 steps) and runs on a laptop CPU in about half a minute. The numbers can differ a little with another PyTorch version.

The script only needs PyTorch. The notebook also needs Matplotlib and Jupyter:

```bash
pip install notebook
jupyter notebook notebook.ipynb
```

Running the notebook saves its figures to an `assets/` folder next to it.

## What to expect

- The loss drops quickly and then flattens at about 2.2. It cannot reach zero: the same point and time appear with different target arrows, and that disagreement stays in the loss.
- With 50 Euler steps, points from the noise first move towards the middle of the data and then spread onto the spiral.
