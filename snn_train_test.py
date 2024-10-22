import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.ao.nn.quantized.functional import threshold
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools

from pyJoules.device import DeviceFactory
from pyJoules.energy_meter import measure_energy
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.handler.pandas_handler import PandasHandler
from pyJoules.energy_meter import EnergyMeter

csv_handler = CSVHandler('result.csv')
pd_handler = PandasHandler()
domains = [NvidiaGPUDomain(0)]
try:
    devices = DeviceFactory.create_devices(domains)
except:
    try:
        devices = DeviceFactory.create_devices([RaplPackageDomain(0)])
    except:
        exit("This code must be run on either a CUDA GPU (linux & windows) or an Intel CPU (on linux)")
meter = EnergyMeter(devices)


# Leaky neuron model, overriding the backward pass with a custom function
class LeakySurrogate(nn.Module):
    def __init__(self, beta, threshold=1.66):
        super(LeakySurrogate, self).__init__()

        # initialize decay rate beta and threshold
        self.beta = beta
        self.threshold = threshold
        self.spike_gradient = self.ATan.apply

    # the forward function is called each time we call Leaky
    def forward(self, input_, mem):
        spk = self.spike_gradient((mem-self.threshold))  # call the Heaviside function
        reset = (self.beta * spk * self.threshold).detach()  # remove reset from computational graph
        mem = self.beta * mem + input_ - reset  # Eq (1)
        return spk, mem

    # Forward pass: Heaviside function
    # Backward pass: Override Dirac Delta with the derivative of the ArcTan function
    @staticmethod
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem):
            spk = (mem > 0).float() # Heaviside on the forward pass: Eq(2)
            ctx.save_for_backward(mem)  # store the membrane for use in the backward pass
            return spk

        @staticmethod
        def backward(ctx, grad_output):
            (spk,) = ctx.saved_tensors  # retrieve the membrane potential
            grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output # Eqn 5
            return grad


# Init neurons
lif1 = LeakySurrogate(beta=0.9)

# dataloader arguments
batch_size = 256
data_path = '/tmp/data/mnist'

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# Network Architecture
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

# Temporal Dynamics
num_steps = 25
beta = 0.95
threshold = 1.0


# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for forward_step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


# Load the network onto CUDA if available
net = Net().to(device)


# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target


def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")


def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")


loss = nn.CrossEntropyLoss()  # TODO Replace this with a fitness function perhaps
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

num_epochs = 3
loss_hist = []
test_loss_hist = []
counter = 0


# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        meter.start(tag=f'ITER{iter_counter}')
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())
        #ctx.record(tag=f'EPOCH_{iter_counter}')


        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net(test_data.view(batch_size, -1))

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                train_printer()
            counter += 1
            iter_counter += 1

        meter.stop()
        data = meter.get_trace()[0]
        print(data.timestamp, data.tag, data.energy)
        # Use energy & loss as a fitness criterion, google turning loss into fitness (check lec slides 2 or 3?)

# For fitness?

total = 0
correct = 0

# drop_last switched to False to keep all samples
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
    net.eval()
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        test_spk, _ = net(data.view(data.size(0), -1))

        # calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")

# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# Network Saturation
# In later epochs, it's possible that the network is becoming saturated, where most of the neurons are highly responsive
# due to over-potentiation of synapses. This can lead to a significant increase in firing activity if
# the synaptic strengths are not properly regularized or constrained. Therefore we will use GA to manage STDP
