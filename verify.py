from hugchat import hugchat
from hugchat.login import Login
from pyecore.resources import ResourceSet, URI
import os

from pathlib import Path
from datasets import load_dataset
import pandas as pd
from datasets import Dataset,DatasetDict, load_dataset
from huggingface_hub import CommitScheduler
from uuid import uuid4
import json
from datetime import datetime


import os
JSON_DATASET_DIR = Path("json_dataset")
JSON_DATASET_DIR.mkdir(parents=True, exist_ok=True)

JSON_DATASET_PATH = JSON_DATASET_DIR / f"train-{uuid4()}.json"

scheduler = CommitScheduler(
    repo_id="VeryMadSoul/errors",
    repo_type="dataset",
    folder_path=JSON_DATASET_DIR,
    path_in_repo="data",
)

def save_json(model: str, errors: list) -> None:
    with scheduler.lock:
        with JSON_DATASET_PATH.open("a") as f:
            json.dump({"model": model, "error": errors, "datetime": datetime.now().isoformat()}, f)
            f.write("\n")

# Log into huggingface and grant authorization to huggingchat
EMAIL = os.environ['HF_EMAIL'] 
PASSWD = os.environ['HF_PASSWORD'] 
cookie_path_dir = "./cookies/" # NOTE: trailing slash (/) is required to avoid errors
sign = Login(EMAIL, PASSWD)
cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)

# Create your ChatBot
chatbot = hugchat.ChatBot(cookies=cookies.get_dict(), system_prompt = '''You are a systems engineer, expert in model driven engineering and meta-modeling
Your OUTPUT should always follow this format :

```XML

< YOUR CODE HERE >

```
''')  # or cookie_path="usercookies/<email>.json"
chatbot.switch_llm(1)

# Create a new conversation
chatbot.new_conversation(switch_to = True) # switch to the new conversation

#create prompt
NLD= '''SimplePDL is an experimental language for specifying processes. The SPEM standard (Software Process Engineering Metamodel) proposed by the OMG inspired our work, but we also took ideas from the UMA metamodel (Unified Method Architecture) used in the EPF Eclipse plug-in (Eclipse Process Framework), dedicated to process modeling. SimplePDL is simplified to keep the presentation simple.
Its metamodel is given in the figure 1. It defines the process concept (Process) composed of a set of work definitions (WorkDefinition) representing the activities to be performed during the development. One workdefinition may depend upon another (WorkSequence). In such a case, an ordering constraint (linkType) on the second workdefinition is specified, using the enumeration WorkSequenceType. For example, linking two workdefinitions wd1 and wd2 by a precedence relation of kind finishToStart means that wd2 can be started only if wd1 is finished (and respectively for startToStart, startToFinish and finishToFinish). SimplePDL does also allow to explicitly represent resources (Resource) that are needed in order to perform one workdefinition (designer, computer, server...) and also time constraints (min_time and max_time on WorkDefinition and Process) to specify the minimum (resp. maximum) time allowed to perform the workdefinition or the whole process.'''
description='''# Writing Ecore Files

## Introduction

Ecore is the core meta-model of the Eclipse Modeling Framework (EMF), which provides a foundation for building tools and applications based on models. Ecore files define the structure and constraints of a model using an XML-based syntax. This document explains how to write Ecore files, covering both the syntax and semantics.

## Syntax

An Ecore file is an XML document that follows the Ecore XML Schema. The root element of an Ecore file is `EPackage`, which represents a package or namespace for the model.

### EPackage

The `EPackage` element has the following attributes:

- `name`: The name of the package.
- `nsURI`: The namespace Uniform Resource Identifier (URI) for the package, which should be a globally unique identifier.
- `nsPrefix`: The preferred namespace prefix to be used for the package.

Example:

```xml
<ecore:EPackage name="library" nsURI="http://example.com/library" nsPrefix="lib">

</ecore:EPackage>
```

### EClassifiers

Inside the `EPackage`, you can define `EClassifiers`, which represent classes, data types, or other types in the model. The two most common types of `EClassifiers` are `EClass` and `EDataType`.

#### EClass

An `EClass` represents a class in the model. It can contain `EStructuralFeatures`, which define the attributes and references of the class.

Example:

```xml
<eClassifiers xsi:type="ecore:EClass" name="Book">
  <eStructuralFeatures xsi:type="ecore:EAttribute" name="title" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EString"/>
  <eStructuralFeatures xsi:type="ecore:EReference" name="author" lowerBound="1" eType="#//Author" containment="true"/>
</eClassifiers>
```

In this example, the `Book` class has a `title` attribute (of type `EString`) and an `author` reference (to the `Author` class).

#### EStructuralFeatures

`EStructuralFeatures` represent the attributes and references of an `EClass`. They can be of type `EAttribute` or `EReference`.

- `EAttribute`: Represents an attribute of the class, with a name and a data type (`eType`).
- `EReference`: Represents a reference to another `EClass`, with a name, a reference type (`eType`), and additional properties like `containment`, `lowerBound`, and `upperBound`.

#### EDataType

An `EDataType` represents a data type in the model, such as `EString`, `EInt`, or a user-defined data type.

Example:

```xml
<eClassifiers xsi:type="ecore:EDataType" name="SSN" instanceClassName="java.lang.String">
  <eAnnotations source="http://www.eclipse.org/emf/2002/GenModel">
    <details key="documentation" value="A social security number"/>
  </eAnnotations>
</eClassifiers>
```

In this example, the `SSN` data type is defined as a string with additional documentation.

### Annotations

Ecore supports annotations, which provide additional metadata or constraints for model elements. Annotations are represented by the `eAnnotations` element.

Example:

```xml
<eAnnotations source="http://www.eclipse.org/emf/2002/GenModel">
  <details key="documentation" value="A social security number"/>
</eAnnotations>
```
### Comments and Documentation
XML comments are not supported in XMI Format
USE EANNOTATIONS instead 



## Semantics

Beyond the syntax, Ecore files also define the semantics of the model, which determine the behavior and constraints of the model elements.

### Inheritance

Ecore supports inheritance between classes. An `EClass` can inherit from one or more `EClasses` using the `eSuperTypes` element.

Example:

```xml
<eClassifiers xsi:type="ecore:EClass" name="FictionBook" eSuperTypes="#//Book">
  <eStructuralFeatures xsi:type="ecore:EAttribute" name="genre" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EString"/>
</eClassifiers>
```

In this example, the `FictionBook` class inherits from the `Book` class and adds a `genre` attribute.

### Containment

Ecore supports containment relationships between classes, which determine ownership and lifecycle management. A containment reference is specified using the `containment="true"` attribute on an `EReference`.

Example:

```xml
<eStructuralFeatures xsi:type="ecore:EReference" name="author" lowerBound="1" eType="#//Author" containment="true"/>
```

In this example, the `author` reference of the `Book` class has containment set to `true`, meaning that the `Author` instance is owned by the `Book` instance.

### Multiplicities

Ecore supports multiplicity constraints on attributes and references, which define the allowed number of values for a feature. Multiplicities are specified using the `lowerBound` and `upperBound` attributes on `EStructuralFeatures`.

Example:

```xml
<eStructuralFeatures xsi:type="ecore:EReference" name="chapters" upperBound="-1" eType="#//Chapter" containment="true"/>
```

In this example, the `chapters` reference of a `Book` class has an unbounded upper bound (`-1`), allowing any number of `Chapter` instances to be contained within the `Book`.

### Operations and Constraints

Ecore also supports the definition of operations and constraints on model elements, although these are typically specified using additional tools or languages, such as the Object Constraint Language (OCL) or Java code.


##Reaccuring errors : 
Invalid tag name error is linked to the tag <!-- --> don't use it in the syntax

## Conclusion

Ecore files provide a structured way to define models using an XML-based syntax. By understanding the syntax and semantics of Ecore files, developers can create robust and well-defined models that can be used as the foundation for various tools and applications within the Eclipse Modeling Framework.'''
#prompt= "Convert the following description into an ecore xmi representation:\n" + NLD  + "\n Here's a technical document of how to write correct ecore file:\n" + description     #WHen tryin to add the description


# Non stream response
#query_result0 = chatbot.chat(prompt)
#print(query_result0) # or query_result.text or query_result["text"]

'''
# Stream response
for resp in chatbot.query(
    "Hello",
    stream=True
):
    print(resp)

# Web search (new feature)
query_result = chatbot.query("Hi!", web_search=True)
print(query_result)
for source in query_result.web_search_sources:
    print(source.link)
    print(source.title)
    print(source.hostname)
'''

# Create a new conversation
#chatbot.new_conversation(switch_to = True) # switch to the new conversation

# Get conversations on the server that are not from the current session (all your conversations in huggingchat)
#conversation_list = chatbot.get_remote_conversations(replace_conversation_list=True)
# Get conversation list(local)
#conversation_list = chatbot.get_conversation_list()

# Get the available models (not hardcore)
#models = chatbot.get_available_llm_models()

# Switch model with given index

#chatbot.switch_llm(2) # Switch to the second model

# Get information about the current conversation
#info = chatbot.get_conversation_info()
#print(info.id, info.title, info.model, info.system_prompt, info.history)

### Assistant
#assistant = chatbot.search_assistant(assistant_name="ChatGpt") # assistant name list in https://huggingface.co/chat/assistants
#assistant_list = chatbot.get_assistant_list_by_page(page=0)
#chatbot.new_conversation(assistant=assistant, switch_to=True) # create a new conversation with assistant
def initial_prompt(NLD, description):
    prompt= "Convert the following description into an ecore xmi representation:\n" + NLD  + "\n Here's a technical document of how to write correct ecore file:\n" + description     #WHen tryin to add the description

    return chatbot.chat(prompt)

def fix_err(xmi, err):
  prompt="Fix the following error: " +str(err)+"\n in the following xmi  :\n" + xmi+ "\n Here's a technical document of how to write correct ecore file:\n" + description

  return chatbot.chat(prompt)

def verify_xmi(output,output_file_name):
  #here we're gonna verify our Model's output by using the either a tool or a developped solution XMI parser

  #Return can be either bool or preferably the actual compilation error or xmi line error
  output = str(output)
  #Returning a bool won't be that helpful ..
  with open("outs\output"+output_file_name+".ecore", "w") as file1:
    # Writing data to a file
    if "```xml" in output:
      file1.writelines(output[output.find("```xml")+len("```xml\n"):output.rfind("```")])
    else:
      file1.writelines(output[output.find("```")+len("```\n"):output.rfind("```")])
  try:
    rset = ResourceSet()
    resource = rset.get_resource(URI("outs\output"+output_file_name+".ecore"))

  except Exception as e:
    return e.args[0]
  return 'no e'

def iterative_prompting(NLD, XMI,max_iter=3):

  history= []

  i=0
    
 
  XMI=""
  output = initial_prompt(NLD, description)
  history.append((NLD,str(output)))
  print(output)

  correct_syntax= verify_xmi(output,str(i))
  errors =[]
  error = (correct_syntax == 'no e')
  errors.append(correct_syntax)

  while (not error) and i<=max_iter:
    i+=1
   

    #print('****************************************')
    #print('Iteration ' +  str(i))
    #print('****************************************')


    error = "\n This Xmi was incorrect. Please fix the errors." + " "+str(correct_syntax)
    
    #print("**************************")
    #print(correct_syntax)
    #print("**************************")


    output = fix_err(output , correct_syntax)
    history.append((error,str(output)))
    #print(output)
    correct_syntax = verify_xmi(output,str(i))
    #print(correct_syntax)
    error = (correct_syntax == 'no e')
    errors.append(correct_syntax)
  
  save_json(chatbot.get_conversation_info().model, errors)

  return history, errors
