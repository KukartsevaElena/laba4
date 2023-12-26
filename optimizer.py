from clearml import Task, Logger
from clearml.automation import (
DiscreteParameterRange, HyperParameterOptimizer,
RandomSearch, UniformIntegerParameterRange
)

task = Task.init(project_name='lab4',
                task_name='optimize_HP',
                task_type=Task.TaskTypes.optimizer,
                reuse_last_task_id=False)

args = {
        'template_task_id': "2408b0fcec63468a93538a47f6b4aeda",
        'run_as_service': False,
}

an_optimizer = HyperParameterOptimizer(
    base_task_id=args['template_task_id'],
    hyper_parameters=[
    UniformIntegerParameterRange('General/max_depth', min_value=3, max_value=9, step_size=3),
    DiscreteParameterRange('General/random_state', values=[0, 1, 5, 15]),
    ],
    objective_metric_title='F1',
    objective_metric_series='f1_score',
    objective_metric_sign='max',
)

an_optimizer.start()
top_exp = an_optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
task.upload_artifact('top_exp', top_exp)
an_optimizer.wait()
an_optimizer.stop()