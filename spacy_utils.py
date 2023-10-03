import spacy
import scispacy
from spacy.language import Language
from scispacy.linking import EntityLinker

class Ner():
    
    def __init__(self, spacy_model: str, pipeline_linker: str, resolve_abbreviations: bool, 
                 ontology: str, threshold: str, max_entities_per_mention: str):
        
        self.spacy_model = spacy_model
        self.resolve_abbreviations = resolve_abbreviations
        self.ontology = ontology
        self.threshold = threshold
        self.max_entities_per_mention = max_entities_per_mention
        self.pipeline_linker = pipeline_linker
        self.nlp = spacy.load(self.spacy_model)
        self.nlp.add_pipe(self.pipeline_linker, 
                          config={"resolve_abbreviations": self.resolve_abbreviations, 
                                "linker_name":self.ontology, "threshold":self.threshold, 
                                "max_entities_per_mention": self.max_entities_per_mention})


        '''
        El método detected_entities aplica la pipeline especificada en la clase Ner, 
        básicamente hace Named Entity Recognition con el modelo de spacy indicado y 
        su componente de NER según la configuración
        '''

    def detected_entities(self, text: str):
        doc = self.nlp(text)
        return doc
        
    '''
        El método linked_entities nos devuelve las entidades directamente conectadas
        a las entidades detectadas según la pipeline de spacy utilizada, a partir de este
        punto ya no hace falta utilizar más la pipeline, sino que guardaremos los resultados 
        obtenidos aquí y los daremos como entrada de datos a la clase entity que carga el
        entity linker. De esta forma evitamos tener en memoria al mismo tiempo la pipeline 
        y el entity linker, lo cual en el caso de un portátil resulta excesiva memoria.
    '''

    def linked_entities(self, doc):
        list_linked_entities = []
        for ent in doc.ents:
            linked_entities = ent._.kb_ents 
            if len(linked_entities) > 0:
                list_linked_entities.append([str(ent), linked_entities])
        return list_linked_entities
        