export interface ForwardRequest {
  question: string;
  top_k?: number;
  similarity_threshold?: number;
  use_rewrite: boolean;
  source_title?: string | null;
}

export interface RetrievedDocMeta {
  start_sec?: number | string | null;
  end_sec?: number | string | null;
  source_url?: string | null;
  title?: string | null;
  source_file_name?: string | null;
  [k: string]: unknown;
}

export interface RetrievedDoc {
  text: string;
  metadata: RetrievedDocMeta;
}

export interface ForwardResponse {
  answer: string;
  context?: string;
  retrieved_documents: RetrievedDoc[];
  retrieval_query?: string;
  rewrite_applied?: boolean;
  source_title?: string | null;
  trace_id?: string | null;
}

export interface SourceFilesResponse {
  files: string[];
}

export interface AuthCheckResponse {
  gate: boolean;
  ok: boolean;
}
