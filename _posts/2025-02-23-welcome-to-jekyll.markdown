---
layout: post
title:  "The Gradio Python Library"
date:   2025-02-23 22:14:47 +0530
categories: jekyll update
---

<p align="center">
  <img src="{{ site.baseurl }}/assets/gradio-icon.png" alt="Alt Text" width="60%">
</p>




##  📄 Introduction

Gradio is a powerful Python library that lets users quickly create interactive web-based interfaces for your machine learning models or any Python functions. It makes it easy to test, showcase, and share your models with others, while also supporting a variety of input and output types like text, images, and audio. Gradio allows for real-time interaction, with live input-output updates, making it easy to see instant feedback. You can also customize the interface to make it visually appealing and integrate it seamlessly with popular machine learning libraries like TensorFlow and PyTorch. Plus, it offers simple deployment, letting you share your work with a broader audience effortlessly.  

Gradio interfaces can be easily shared, either by hosting them locally or using third party software such as Huggingface Spaces. Gradio apps can also be integrated with external applications using its API endpoint. This allows Gradio apps to be integrated with websites and other apps to showcase Machine learning models or any python functions.

## ⚙️ Installation Guide

# For Windows:
Gradio can be installed using pip in your command prompt, through the following lines:

```
pip install gradio
```

or 

```
pip install --upgrade gradio
```

This installs the recent, most stable version of gradio, if you want to install a specific version use the command:
```
pip install gradio==X.X.X 
```
where `X.X.X` is the version of gradio being installed.

While installing, make sure the version of Gradio being installed is compatible with the python version installed on your device. The compatibility of Gradio with different versions of python is given below:

<style>
  table {
    width: 100%;
    border-collapse: collapse;
  }
  th, td {
    border: 1px solid #ddd;
    padding: 10px;
    text-align: center;
    width: 50%;
  }
  th {
    background-color: #f4f4f4;
  }
</style>

| Gradio Version | Required Python Version |
|---------------|----------------------|
| Gradio 5.x    | Python 3.8+          |
| Gradio 4.x    | Python 3.7+          |
| Gradio 3.x    | Python 3.7+          |
| Gradio 2.x    | Python 3.6+          |
| Gradio 1.x    | Python 3.6+          |
| Gradio 0.x    | Python 3.5+          |

# For Linux:
To install gradio on some common linux distributions, we can firstly install pip separately and then use that to install Gradio.

```
sudo apt update && sudo apt install python3 python3-pip -y   # Ubuntu/Debian  
sudo yum install python3 python3-pip -y                      # CentOS/RHEL  
```
After this, to install Gradio:

```
pip install gradio
```

# For Mac

To install Gradio on mac, open a terminal application using `Command (⌘) + Space` and run the following command:
```
pip3 install gradio

```


## 🔑 Key Features
Gradio has many important features and tools to provide the user with all means necessary to demonstrate their models or functions. Some of the key features of this library are:

# 1. Interfaces

The code snippet for the following is:
```
import gradio as gr

def calculator(num1, operation, num2 ):
    global message
    if operation == "add":
        message= num1 + num2
    elif operation == "subtract":
        message= num1 - num2
    elif operation == "multiply":
        message= num1 * num2
    elif operation == "divide":
        if num2 == 0:
            message="Cannot divide by zero"
        else:
            message=num1 / num2
    return message

def store_message(num1,num2,operation, history: list[str]):  
    output = {
        "Result": calculator(num1,operation,num2),
        "History": history[::-1]
    }
    message=calculator(num1,operation,num2)
    history.append(f"{num1} {operation} {num2} = {message}")
    return output, history

demo = gr.Interface(fn=store_message,
                    inputs=["number",gr.Slider(value=0,minimum=-100,maximum=100),gr.Radio(["add", "subtract", "multiply", "divide"]), gr.State(value=[])],
                    outputs=["json", gr.State()],
                    title=" Toy Calculator with History",
                    description="We can save the data being executed in the toy calculator through state interfaces"
                    )

demo.launch()
```
<iframe src="https://abhisheknair-24110009-gradio-app-1.hf.space" width="100%" height="600px" style="border:none;"></iframe> 

# 2. Blocks

# 3. Advanced Blocks

The code snippet for the following is:
```
import gradio as gr

with gr.Blocks() as demo:

    tasks = gr.State([])  # Store tasks
    new_task = gr.Textbox(label="Task Name", autofocus=True)

    # Function to add a new task
    def add_task(tasks, new_task_name):
        return tasks + [{"name": new_task_name, "complete": False, "deleted": False}], ""

    # Submit button to add task
    new_task.submit(add_task, [tasks, new_task], [tasks, new_task])

    # Function to render tasks
    @gr.render(inputs=tasks)
    def render_todos(task_list):
        # Categorizing tasks into complete, incomplete, and deleted
        complete = [task for task in task_list if task["complete"]]
        incomplete = [task for task in task_list if not task["complete"] and not task["deleted"]]
        deleted = [task for task in task_list if task["deleted"]]

        # Rendering Incomplete Tasks
        gr.Markdown(f"### Incomplete Tasks ({len(incomplete)})")
        for task in incomplete:
            with gr.Row():
                gr.Textbox(task['name'], show_label=False, container=False)
                done_btn = gr.Button("Done", scale=0)
                
                # Mark task as complete
                def mark_done(task=task):
                    task["complete"] = True
                    return task_list
                
                done_btn.click(mark_done, None, [tasks])

                delete_btn = gr.Button("Delete", scale=0, variant="stop")
                
                # Mark task as deleted
                def delete(task=task):
                    task["deleted"] = True
                    return task_list
                
                delete_btn.click(delete, None, [tasks])

        # Rendering Complete Tasks
        gr.Markdown(f"### Complete Tasks ({len(complete)})")
        for task in complete:
            gr.Textbox(task['name'], show_label=False, container=False)

        # Rendering Deleted Tasks
        gr.Markdown(f"### Deleted Tasks ({len(deleted)})")
        for task in deleted:
            gr.Textbox(task['name'], show_label=False, container=False)

# Launch the Gradio interface
demo.launch()
```
<iframe src="https://abhisheknair-24110009-gradio-app-3.hf.space" width="100%" height="600px" style="border:none;"></iframe> 

# 4. Chatbot

The code snippet for the following is:
```
import gradio as gr

# Function to generate chatbot response
def chatbot(input_text, chat_history):
    # Simple example: respond by reversing the input text
    response = f"Bot: {input_text[::-1]}"
    
    # Append both user input and bot response to chat history
    chat_history.append((input_text, response))
    
    # Return updated history and the current response
    return chat_history, response

# Set up the Gradio interface
with gr.Blocks() as demo:
    # Define the state to store the conversation history
    chat_history = gr.State([])
    
    # User input component
    user_input = gr.Textbox(label="Your Message", placeholder="Type here...")
    
    # Bot response component
    bot_response = gr.Textbox(label="Bot Response", interactive=True)
    
    # Send button to trigger the chatbot response
    submit_button = gr.Button("Send")
    
    # The button click triggers the chatbot function with user input and history
    submit_button.click(
        chatbot, 
        inputs=[user_input, chat_history], 
        outputs=[chat_history, bot_response]
    )
# Launch the Gradio interface
demo.launch()
```
<iframe src="https://abhisheknair-24110009-gradio-app-2.hf.space" width="100%" height="600px" style="border:none;"></iframe> 
