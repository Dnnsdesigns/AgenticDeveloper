import gradio as gr
import tensorflow as tf
import numpy as np
from datasets import load_dataset
from network import create_text_neural_network, create_gating_network
from agent import PrimeAgent, SecondaryAgent

# Verify GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define the model training and evaluation function
def train_and_test_model(epochs, batch_size):
    vocab_size = 10000
    embedding_dim = 128
    input_length = 100
    num_classes = 10
    num_experts = 3  # Number of experts

    # Create models for the gating network and secondary agents
    gating_network = create_gating_network((input_length,), num_experts)
    expert_networks = [create_text_neural_network(vocab_size, embedding_dim, input_length, num_classes) for _ in range(num_experts)]
    
    # Define specialties for secondary agents
    specialties = ['code writing', 'code debugging', 'code optimization']

    # Create secondary agents
    secondary_agents = [SecondaryAgent(expert_networks[i], specialties[i]) for i in range(num_experts)]

    # Create prime agent with secondary agents
    prime_agent = PrimeAgent(gating_network, secondary_agents)

    # Load dataset using Hugging Face datasets library
    dataset = load_dataset('imdb')
    train_data = np.array([example['text'][:input_length] for example in dataset['train']])
    train_labels = np.array([example['label'] for example in dataset['train']])
    test_data = np.array([example['text'][:input_length] for example in dataset['test']])
    test_labels = np.array([example['label'] for example in dataset['test']])

    # Convert text to numerical data (tokenization)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(train_data)
    train_data = tokenizer.texts_to_sequences(train_data)
    test_data = tokenizer.texts_to_sequences(test_data)
    train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=input_length)
    test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=input_length)

    # Train and test the prime agent's model
    results = ""
    with tf.device('/GPU:0'):
        prime_agent.gating_network.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)
        test_loss, test_acc = prime_agent.gating_network.evaluate(test_data, test_labels, verbose=2)
        results += f'Gating Network Test Accuracy: {test_acc}\\n'
        
        for expert in prime_agent.experts:
            expert.model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)
            test_loss, test_acc = expert.model.evaluate(test_data, test_labels, verbose=2)
            results += f'{expert.specialty.capitalize()} Expert Test Accuracy: {test_acc}\\n'

    return results

# Define the Gradio interface
gr_interface = gr.Interface(
    fn=train_and_test_model,
    inputs=[
        gr.inputs.Slider(minimum=1, maximum=50, step=1, default=10, label="Epochs"),
        gr.inputs.Slider(minimum=16, maximum=512, step=16, default=128, label="Batch Size")
    ],
    outputs="text",
    title="Developer Assistant Training Interface",
    description="Adjust the training parameters and train the model."
)

# Launch the interface
gr_interface.launch()
