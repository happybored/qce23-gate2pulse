# from qiskit_ibm_provider import IBMProvider
from qiskit import IBMQ

IBMQ.delete_account()
token = ''

IBMQ.save_account(token)

IBMQ.load_account() # Load account from disk