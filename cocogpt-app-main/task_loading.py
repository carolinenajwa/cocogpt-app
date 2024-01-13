import json

defined_tasks_file = 'task/COCO_task.json'

def load_a_predefined_task(task_file_path: str):
    task = {
        'task_name': '',
        'overview': '',
        'goal': '',
        'checklist': [],
        'subtasks': []
    }
    with open(task_file_path, 'r') as f:
        obj = json.load(f)
        task['task_name'] = obj['task_name']
        task['overview'] = obj['overview']
        task['goal'] = obj['goal']
        task['checklist'] = obj['checklist']
        task['subtasks'] = obj['subtasks']

    # Checkpoint: Output loaded task for verification
    # print("Loaded task:", task)

    return task

def load_a_predefined_task_by_file_name(task_file_path: str):
    return load_a_predefined_task(task_file_path)

def load_predefined_tasks():
    task_list = [load_a_predefined_task_by_file_name(defined_tasks_file)]
    tasks = {}
    for task in task_list:
        tasks[task['task_name']] = {
            'overview': task['overview'],
            'goal': task['goal'],
            'checklist': task['checklist'],
            'subtasks': task['subtasks'],
        }

    # Checkpoint: Output all loaded tasks
    # print("All loaded tasks:", tasks)

    return tasks

if __name__ == '__main__':
    # Load and print tasks
    predefined_tasks = load_predefined_tasks()
    print("Predefined tasks:", predefined_tasks)