"""This is the execution script. According to the chosen mode it will run the train,
 the evaluation or the inference module.
To use the script run:
python execute.py /
 --experiment_name= --experiments_dir= --config_file=  /
 --mode=train
"""
import logging

from absl import app, flags

from core.data.data_loader import get_dataset
from core.data.data_visualization import Visualization
from core.utils.configs import get_config_from_yaml
from core.utils.exp_utils import create_experiment
from core.workflows.evaluate import evaluation_pipeline
from core.workflows.inference import inference_pipeline
from core.workflows.train import train

flags.DEFINE_string("experiment_name", None, "name of your experiment")
flags.DEFINE_string("experiments_dir", None, "path to the experiments directory")
flags.DEFINE_string("config_file", None, "path of your config file")
flags.DEFINE_string("model", None, "name of the pre-trained model to use")

flags.DEFINE_string("mode", "train", "mode to run: train, eval or inference")
flags.DEFINE_boolean("visualize", False, "Visualize a batch from initial data")
flags.DEFINE_float("iou_threshold", 0.5, "The intersection over union threshold")
flags.DEFINE_float(
    "score_threshold", 0.5, "The score threshold to filter the bounding boxes"
)
flags.DEFINE_string("experiment_path", None, "path of your experiment")
flags.DEFINE_string("image_dir", None, "the directory containing the test images")
flags.DEFINE_string(
    "annot_file_path", None, "the path of the annotation file for evaluation"
)


FLAGS = flags.FLAGS


def main(argv):
    """
    According to the chosen mode it will run the train,
    the evaluation or the inference module.
    """
    # pylint: disable=unused-argument
    flags.mark_flag_as_required("experiment_name")
    flags.mark_flag_as_required("experiments_dir")
    flags.mark_flag_as_required("config_file")

    experiment_name = FLAGS.experiment_name
    experiments_dir = FLAGS.experiments_dir
    config_file = FLAGS.config_file
    model = FLAGS.model
    visualize = FLAGS.visualize

    iou_threshold = FLAGS.iou_threshold
    score_threshold = FLAGS.score_threshold
    experiment_path = FLAGS.experiment_path
    image_dir = FLAGS.image_dir
    annot_file_path = FLAGS.annot_file_path

    mode = FLAGS.mode
    if visualize:
        config = get_config_from_yaml(config_file)
        # Getting the batched and augmented dataset
        batched_dataset = get_dataset(config=config, phase=mode)
        visualizer = Visualization(config=config, phase=mode)
        # View one batch of the augmented dataset
        visualizer.view_image(dataset=batched_dataset, block=True)
        # View one batch of the dataset with the labels and bounding boxes
        visualizer.visualize_annotated_objects(dataset=batched_dataset, block=True)
    if mode == "train":

        experiment_config_path = create_experiment(
            experiment_name, experiments_dir, config_file, model
        )
        experiment_config = get_config_from_yaml(experiment_config_path)

        if experiment_config:
            train(experiment_config_path)

    elif mode == "eval":

        evaluation_pipeline(
            experiment_path=experiment_path,
            image_dir=image_dir,
            annot_file_path=annot_file_path,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )

    elif mode == "inference":

        inference_pipeline(
            experiment_path=experiment_path,
            image_dir=image_dir,
            score_threshold=score_threshold,
        )

    else:
        logging.error("Please specify a valid mode: train, eval or inference")


if __name__ == "__main__":
    app.run(main)
