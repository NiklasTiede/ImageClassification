""" Tools helping to record data of training run, to facilitate the
 comparison of different hyperparameters """
import json
import time
from collections import namedtuple
from collections import OrderedDict
from itertools import product
from typing import List, Any, Union, Optional
import pandas as pd
import torch
import torchvision
from IPython.display import clear_output
from IPython.display import display
from torch.utils.tensorboard import SummaryWriter
import model
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

start_time = time.time()


# how to use only GPU for training?  -> still to be implemented
# use_gpu = True if torch.cuda.is_available() else False  # rest of the code?


class RunManager():
    def __init__(self) -> None:
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data: List[OrderedDict[str, Union[int, float]]] = []  # List[OrderedDict[str, Union[int, float]]]
        self.run_start_time: Optional[float] = None

        self.network: Optional[model.Network] = None
        self.loader = None
        self.tb = None

    def begin_run(self, run: Any, network: model.Network, loader: Any) -> None:

        self.run_start_time = time.time()
        # reveal_type(loader)
        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)

    def end_run(self) -> None:
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self) -> None:
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self) -> None:

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()  # saves data to ourselves
        print('results type:', type(results))
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        clear_output(wait=True)
        display(df)

    def track_loss(self, loss) -> None:
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_num_correct(self, preds, labels) -> None:
        self.epoch_num_correct += self.get_num_correct(preds, labels)

    @torch.no_grad()
    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName) -> None:

        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)


class RunBuilder():  # iterating hyperparameters
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        # reveal_type(runs)
        # print('runs type:', type(runs))
        return runs
