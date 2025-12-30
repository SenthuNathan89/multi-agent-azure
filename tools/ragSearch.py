from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from langchain_core.documents import Document

class AzureSearchVector:
    def __init__(self, endpoint, key, index_name, embeddings, vector_field="contentVector", text_field="content"):
        # Clean endpoint
        endpoint = endpoint.rstrip('/')
        if '/indexes/' in endpoint:
            endpoint = endpoint.split('/indexes/')[0]
        
        # Create credential properly
        credential = AzureKeyCredential(key)
        try:
            self.client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=credential  
            )
            self.embeddings = embeddings
            self.vector_field = vector_field
            self.text_field = text_field
            
            # Test connection
            # print(f"  Testing connection...")
            
            # Try to get document count
            results = self.client.search(search_text="*", top=1, include_total_count=True)
            print(f"Connected successfully to Azure Search")
            
        except Exception as e:
            print(f"Failed to initialize Azure Search client")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 3):
        try:
            print(f"Performing similarity search for: '{query}'")
            
            # Generate embedding
            query_vector = self.embeddings.embed_query(query)
            print(f"Generated embedding vector of length: {len(query_vector)}")
            
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=k,
                fields=self.vector_field
            )
            
            # Search - select all fields to see what's available
            results = self.client.search(
                search_text=None,
                vector_queries=[vector_query],
                top=k
            )
            
            # Convert to documents and collect metadata
            docs = []
            sources_info = []
            
            for i, result in enumerate(results):
                content = result.get(self.text_field, "")
                if content:
                    # print(f" Result {i+1}: {content[:100]}...")
                    # Extract source information - use .get() to avoid KeyError
                    source_data = {
                        "id": result.get("parent_id", "N/A"),
                        "path": result.get("filepath", "N/A"),
                        "title": result.get("title", "N/A")
                    }
                    sources_info.append(source_data)
                    
                    # Create document with metadata
                    docs.append(Document(
                        page_content=content,
                        metadata=source_data
                    ))
            
            print(f"Found {len(docs)} results (but may have multiple differing chunks from the same source file)")
            # Remove duplicates by converting to tuples
            unique_tuples = set(tuple(sorted(d.items())) for d in sources_info)
            sources_info = [dict(t) for t in unique_tuples]
            
            # Print all unique sources at the end
            print("*"*100 +"\n")

            print(f"Sources from Azure AI Search ({len(sources_info)} total):")
            for i, source in enumerate(sources_info, 1):
                print(f"\nSource {i}:")
                print(f"  Document ID: {source.get('id', 'N/A')}")
                print(f"  File Path: {source.get('path', 'N/A')}")
                print(f"  Title: {source.get('title', 'N/A')}")
                print("-"*100 + "\n")
            
            return docs
            
        except Exception as e:
            print(f"Search error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return []
