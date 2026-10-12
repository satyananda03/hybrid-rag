export type GraphProperties = Record<string, unknown>
/** Whatever workspace name the ingest pipeline actually wrote — see getGraphWorkspaces(). */
export type GraphWorkspace = string

export interface KnowledgeGraphNode {
  id: string
  labels: string[]
  properties: GraphProperties
}

export interface KnowledgeGraphEdge {
  id: string
  source: string
  target: string
  type?: string
  properties: GraphProperties
}

export interface KnowledgeGraphResponse {
  nodes: KnowledgeGraphNode[]
  edges: KnowledgeGraphEdge[]
  is_truncated?: boolean
}

export interface GraphQuery {
  label: string
  maxDepth: number
  maxNodes: number
  workspace: GraphWorkspace
}
