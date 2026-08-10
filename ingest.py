
import chromadb

client = chromadb.PersistentClient('chromaDB')
RAGCollection = client.get_or_create_collection(
    name='RAG_Pipeline',
    metadata={"hnsw:space": "cosine"}
)

def ingest(filepath : str, collection: chromadb.Collection):
    with open(filepath, 'r') as file:
        content = file.read()
    
    def sentence_chunks (text: str, sentences_per_chunk : int, )  -> list[str]:
        chunks : list[str] = []
        paragraphs = [p for p in text.split('\n\n') if len(p.strip()) > 0]

        merged_paragraphs = []
        carry_over = ""

        for paragraph in paragraphs:
                combined = carry_over + "\n\n" + paragraph if carry_over else paragraph

                if len(combined.split()) < 8:
                     carry_over = combined   # too short — hold onto it, don't commit it yet
                else:
                    merged_paragraphs.append(combined)
                    carry_over = ""   
        if carry_over : merged_paragraphs.append(carry_over)
        for paragraph in merged_paragraphs:
            splitting = paragraph.split('. ')
            if len(splitting) <= sentences_per_chunk:
                chunks.append(paragraph)
            else:
                for i in range(0 , len(splitting), n := sentences_per_chunk ):
                    chunk = '. '.join(splitting[i : i + n ])
                    chunks.append(chunk)
        return chunks
        
    sChunks = sentence_chunks(content, 2)
    collection.add(
        ids= [f'number {i}' for i in range(1, len(sChunks) + 1)],
        documents = sChunks
    )
    
ingest('RAG/fitness_knowledge_base.txt', RAGCollection)