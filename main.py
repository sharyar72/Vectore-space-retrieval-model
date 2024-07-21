import tkinter as tk
from tkinter import filedialog
import os
import pickle
from preprocessing import get_documents, create_inverted_index
from query import query_processing, calculate_tfidf_document
import time


# global variables
inverted_index={}
document_tfidf={}
folder_path=''

#function to load the indexes from the folder
def load_indexes(folder_path):
    flag=False
    global inverted_index

    file_format = ".pkl" # the file format you want to check

    for file_name in os.listdir(folder_path):
        if file_name.endswith(file_format):
            flag=True
            break
            # or do whatever you want with the file
    
    if flag:
        file_path_1=folder_path+'\\inverted_index.pkl'
        # Load inverted and positional indexes from files
        with open(file_path_1, 'rb') as f:
            inverted_index = pickle.load(f)
    else:
        inverted_index=create_inverted_index(folder_path)


#function to choose folder from the Computer with text files for preprocessing
def choose_folder():
    global folder_path 
    folder_path= filedialog.askdirectory()
    run_function()
    


#function to show that the files are being preprocessed
def run_function():
    loading_window = tk.Toplevel(root)
    loading_window.title("Loading")
    tk.Label(loading_window, text="Loading...").pack()
    loading_window.update()

    # Call the long running function
    load_indexes(folder_path)
    documents=get_documents(folder_path)
    global document_tfidf
    document_tfidf=calculate_tfidf_document(documents,inverted_index,len(documents))

    tk.Label(loading_window, text="Completed").pack()
    loading_window.update()

    # Wait for 2 seconds before closing the window
    time.sleep(2)
    loading_window.destroy()


#function to clear input from screen
def clear_text():
    text_box.delete(0, 'end')


# check query and compute results
def evaluate_query(query_string):
    if ',' in query_string:
        parts = query_string.split(',')
        # Extract the number part after the comma
        alpha = float(parts[1].strip())
        result = query_processing(parts[0],inverted_index,document_tfidf,alpha)
    else:
        result = query_processing(query_string,inverted_index,document_tfidf)
    #return the result of the query
    return result

# function which is called when enter is pressed displays the result of the query entered
def perform_operation():
    input_str = text_box.get()
    result=evaluate_query(input_str)
    # result_print=sorted(result,key=lambda x:int(x))
    if len(result)!=0:
        display_box.config(state='normal')
        display_box.delete(0, 'end')
        display_box.insert(0, result)
        display_box.config(state='disabled')
    else:
        display_box.config(state='normal')
        display_box.delete(0, 'end')
        display_box.insert(0, 'No documents found')
        display_box.config(state='disabled')

#function to exit the code
def exit_program():
    root.destroy()

root = tk.Tk()
root.title("Vector space model")
root.geometry("500x500")
# Create buttons
choose_folder_btn = tk.Button(root, text="Choose Folder", command=choose_folder,bg="blue",fg="black")
clear_btn = tk.Button(root, text="Clear", command=clear_text,bg="green",fg="black")
enter_btn = tk.Button(root, text="Enter", command=perform_operation,bg="yellow",fg="black")
exit_btn = tk.Button(root, text="Exit", command=exit_program, bg="red",fg="black")

# Create input and output widgets
text_box = tk.Entry(root,width=50,bg='white',fg='black')
display_box = tk.Entry(root, state='disabled',width=50,bg='white',fg='black')

# Set layout using grid
choose_folder_btn.grid(row=0, column=0, sticky='w')
clear_btn.grid(row=1, column=0, sticky='w')
enter_btn.grid(row=2, column=0, sticky='w')
exit_btn.grid(row=3, column=0, sticky='w')

text_box.grid(row=0, column=1, columnspan=2, sticky='we')

display_box.grid(row=1, column=1, columnspan=2,sticky='we')
choose_folder_btn.place(x=0,y=40)
clear_btn.place(x=0,y=80)
enter_btn.place(x=0,y=120)
exit_btn.place(x=0,y=160)
# Configure column and row weights
root.columnconfigure(1, weight=1)
root.rowconfigure(1, weight=1)

root.mainloop()
