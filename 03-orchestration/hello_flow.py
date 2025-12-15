from prefect import flow, task

# 1. Define a Task: The smallest unit of work
@task(log_prints=True)
def print_hello(name: str):
    """A simple task that prints a greeting."""
    print(f"Task: Hello, {name}! Starting my flow run.")
    return f"Greeting for {name}"

# 2. Define a Flow: The container that orchestrates tasks
@flow(name="Hello World Flow")
def my_hello_workflow(person_name: str):
    """
    This is the main flow that calls our task.
    """
    greeting_result = print_hello(person_name)
    print(f"Flow: The task returned: {greeting_result}")
    
# 3. Execute the flow like a normal Python function
if __name__ == "__main__":
    my_hello_workflow("Data Scientist")