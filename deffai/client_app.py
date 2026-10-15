"""ClientApp: runs on each SuperNode, trains/evaluates on its local partition."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from deffai.task import build_model, get_device, load_client_data
from deffai.task import test as test_fn
from deffai.task import train as train_fn

app = ClientApp()


def _setup(msg, context):
    cfg = context.run_config
    model = build_model(cfg["dataset"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    trainloader, valloader = load_client_data(
        context.node_config["partition-id"],
        context.node_config["num-partitions"],
        cfg,
    )
    return cfg, model, trainloader, valloader


@app.train()
def train(msg: Message, context: Context):
    cfg, model, trainloader, _ = _setup(msg, context)
    config = msg.content["config"]
    train_loss = train_fn(
        model,
        trainloader,
        cfg["local-epochs"],
        config["lr"],
        get_device(),
        proximal_mu=config.get("proximal-mu", 0.0),  # set by the FedProx strategy
    )
    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
        }),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    _, model, _, valloader = _setup(msg, context)
    loss, acc = test_fn(model, valloader, get_device())
    content = RecordDict({
        "metrics": MetricRecord({
            "eval_loss": loss,
            "eval_acc": acc,
            "num-examples": len(valloader.dataset),
        }),
    })
    return Message(content=content, reply_to=msg)
