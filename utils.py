#from segtok.segmenter import split_single
import gc

'''
Aquí vamos a poner también las funciones que vamos a utilizar para procesar el texto 
previamente al uso de la pipeline de NER: si vamos a separar el texto en oraciones,
eliminar stop words, pasar a minúsculas, etc. Aunque en principio interesa que se realice
NER sobre el abstract completo.
'''

def get_indices_of_string_matches(my_list, target_string):
    return [index for index, item in enumerate(my_list) if target_string in item]

#def split_into_sentences(text):
#    return split_single(text)

def remove_stop_words(text):
    pass

def character_count(string, character):
    counter = 0
    for char in string:
        if char.lower() == character.lower():
            counter += 1

    return counter 

def to_lowercase(text):
    return text.lower()

def garbage_out(garbage:list):
    for item in garbage:
        del item
    gc.collect
    garbage.clear()
