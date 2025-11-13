from azureml.core import Run
import argparse
import traceback
from util.model_helper import get_model

run = Run.get_context()

# if you would like to run this script on a local computer
# the following code is a good starting point for you, 
# use:
# python -m evaluate.evaluate_model
# in diabetes_regression folder context

if run.id.startswith('OfflineRun'):
    from dotenv import load_dotenv
    # For local development, set values in this section
    load_dotenv()
    sources_dir = os.environ.get("SOURCES_DIR_TRAIN")
    if sources_dir is None:
        sources_dir = 'diabetes_regression'
    path_to_util = os.path.join(".", sources_dir, "util")
    sys.path.append(os.path.abspath(path_to_util))
    
    from model_helper import get_model
    workspace_name = os.environ.get("WORKSPACE_NAME")
    experiment_name = os.environ.get("EXPERIMENT_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    tenant_id = os.environ.get("TENANT_ID")
    model_name = os.environ.get("MODEL_NAME")
    app_id = os.environ.get("SP_APP_ID")
    app_secret = os.environ.get("SP_APP_SECRET")
    build_id = os.environ.get("BUILD_BUILDID")
    # run_id useful to query previous runs
    run_id = "57fee47f-5ae8-441c-bc0c-d4c371f32d70"
    aml_workspace = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group
    )
    ws = aml_workspace
    exp = Experiment(ws, experiment_name)

# comment the following three lines
# if you would like to use Offline mode
exp = run.experiment
ws = run.experiment.workspace
run_id = 'amlcompute'

parser = argparse.ArgumentParser("evaluate")
parser.add_argument("--run_id", type=str, help="Training run ID")
parser.add_argument("--model_name", type=str, help="Name of the Model", default="diabetes_model.pkl")
parser.add_argument("--allow_run_cancel", type=str, help="Set this to false to avoid evaluation step from cancelling run after an unsuccessful evaluation", default="true")

args = parser.parse_args()
if args.run_id is not None:
    run_id = args.run_id
    
if run_id == 'amlcompute':
    run_id = run.parent.id
    
model_name = args.model_name
metric_eval = "mse"

allow_run_cancel = args.allow_run_cancel

# Parameterize the matrics on which the models should be compared
# Add golden data set on which all the model performance can be evaluated
try:
    firstRegistration = False
    tag_name = 'experiment_name'
    
    model = get_model(
        model_name=model_name,
        tag_name=tag_name,
        tag_value=exp.name,
        aml_workspace=ws
    )
    
    if model is not None:
        production_model_mse = 10000
        if metric_eval in model.tags:
            production_model_mse = float(model.tags[metric_eval])
            
        try:
            new_model_mse = float(run.parent.get_metrics().get(metric_eval))
        except TypeError:
            new_model_mse = None
            
        if production_model_mse is None or new_model_mse is None:
            print("Unable to find ", metric_eval, " metrics, ", "existing evaluation")
            if allow_run_cancel.lower() == 'true':
                run.parent.cancel()
        else:
            print("Current Production model {}: {}, ".format(metric_eval, production_model_mse) +
                  "New trained model {}: {}".format(metric_eval, new_model_mse))
            
        if new_model_mse < production_model_mse:
            print("New trained model performs better, thus it should be registered")
        else:
            print("New trained model metrics is worse than or equal to production model so skipping model registration.")
            if allow_run_cancel.lower() == 'true':
                run.parent.cancel()
    else:
        print("This is the first model, thus it should be registered")

except Exception:
    traceback.print_exc(limit=None, file=None, chain=True)
    print("Something went worng trying to evaluate. Existing.")
    raise