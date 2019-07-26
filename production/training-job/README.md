# Transformer Fine-Tuning Job


## Usage 

Run the training job by executing this command with your chosen configuration, exposing port 5000 to the outside world:

```bash
docker run -p 5000:5000 finetuning-job
```

To enable the Nvidia Container Runtime for GPU-acceleration, execute:

```bash
docker run --runtime=nvidia finetuning-job
```

Execute this command for interactive run mode:
```bash
docker run -it --entrypoint=/bin/bash finetuning-job
```

More options:

* publish the container's port 5000 via to `host_port` usinng `-p {host_port}:5000`
* for testing purposes (fast execution), run with `--env-file==test.env`.
* run in detached mode usingn `-d`


### Environment variables

The training job can be parametrized with environment variables. These can be defined by passing an [environment file](https://docs.docker.com/compose/compose-file/#env_file) via `--env-file=={env_file}` to `docker run`. To learn about available variables, click [here](#parameters).



### Configuration

#### Parameters

The training job can be configured with following environment variables:

Example varibables:

<table>
    <tr>
        <th>Variable</th>
        <th>Description</th>
        <th>Default</th>
    </tr>
     <tr>
        <td>NUM_TRAIN</td>
        <td>Max no. of training samples.</td>
        <td>5000</td>
    </tr>
    <tr>
        <td>NUM_TEST</td>
        <td>Max no. of test samples.</td>
        <td>5000</td>
    </tr>
    <tr>
        <td>NUM_MAX_POSITIONS</td>
        <td>Max no. of positions to use as context (sequence length, max 256).</td>
        <td>256</td>
    </tr>
    <tr>
        <td>N_EPOCHS</td>
        <td>No. of training epochs</td>
        <td>2</td>
    </tr>  
    <tr>  
        <td>BATCH_SIZE</td>
        <td>Size of training/test batches for SGD.</td>
        <td>32</td>
    </tr>
     <tr>
        <td>VALID_PCT</td>
        <td>Percentage of validation set split from train set.</td>
        <td>0.05</td>
    </tr>    
    <tr>
        <td>LR</td>
        <td>Initial learning rate.</td>
        <td>0.000065</td>
    </tr>
    <tr>
        <td>MAX_NORM</td>
        <td>Max. value of L2 norm of weights.</td>
        <td>1.5</td>
    </tr>
    <tr>
        <td>SEED</td>
        <td>Random seed used to torch and numpy.</td>
        <td>1337</td>
    </tr>
    <tr>
        <td>NVIDIA_VISIBLE_DEVICES</td>
        <td>Controls which GPUs will be accessible inside the container.</td>
        <td>all</td>
    </tr>
    <tr>
        <td>OMP_NUM_THREADS</td>
        <td>No. of OpenMP threads used by PyTorch. Shouldn't exceed the no. of physical threads.</td>
        <td>8</td>
    </tr>
    <tr>
        <td>BODY</td>
        <td>Whether to train the Transformer body. If zero, only train the classifier head (~800 params).</td>
        <td>1</td>
    </tr>

   
</table>

#### Proxy

If a proxy is required, you can pass it via the `http_proxy`and `no_proxy` environment varibales. For example: `--env http_proxy=<server>:<port>`

#### Docker Configuration

You can find more ways of configuration about [docker run](https://docs.docker.com/engine/reference/commandline/run) and [docker service create](https://docs.docker.com/engine/reference/commandline/service_create) in the official Docker documentation.

## Develop

### Requirements

- Python 3, Maven, Docker

### Build

Execute this command in the project root folder to build the docker container:

```bash
python build.py --name finetuning-job --version={MAJOR.MINOR.PATCH-TAG}
```

The version has to be provided. The version format should follow the [Semantic Versioning](https://semver.org/) standard (MAJOR.MINOR.PATCH). For additional script options:

```bash
python build.py --help
```

### Deploy

Execute this command in the project root folder to push the container to the configured docker registry:

```bash
python build.py --name finetuning-job --deploy --version={MAJOR.MINOR.PATCH-TAG}
```

The version has to be provided. The version format should follow the [Semantic Versioning](https://semver.org/) standard (MAJOR.MINOR.PATCH). For additional script options:

```bash
python build.py --help
```

#### Configure Docker Repository

In order to pull and push docker containers, a Docker registry needs to be configured:

```bash
docker login <server>
```

and entering your user and password to login.
