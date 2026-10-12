import axios from 'axios'
import type { GraphQuery, GraphWorkspace, KnowledgeGraphResponse } from './types'

// Relative on purpose: the page is served at `{prefix}/` and the API at
// `{prefix}/api` (see `hybrid_rag.dashboard.mount_dashboard`), so the same build
// works under any prefix. In dev, Vite proxies `/api` to the backend.
const graphApiClient = axios.create({ baseURL: 'api' })

const GRAPH_KEY_PREFIX = 'graph'

/** Key cache SWR untuk satu subgraph. */
export function graphCacheKey({ workspace, label, maxDepth, maxNodes }: GraphQuery, version = 0) {
  return [GRAPH_KEY_PREFIX, workspace ?? null, label.trim() || '*', maxDepth, maxNodes, version] as const
}

/** Fetches the connected subgraph around `label`. Caching belongs to the caller. */
export async function fetchKnowledgeGraph(query: GraphQuery): Promise<KnowledgeGraphResponse> {
  const { data } = await graphApiClient.get<KnowledgeGraphResponse>('/graph', {
    params: { workspace: query.workspace, label: query.label, max_depth: query.maxDepth, max_nodes: query.maxNodes },
  })
  return data
}

export async function getPopularGraphLabels(workspace: GraphWorkspace, limit = 50): Promise<string[]> {
  const { data } = await graphApiClient.get<string[]>('/labels/popular', { params: { workspace, limit } })
  return data
}

export async function searchGraphLabels(workspace: GraphWorkspace, query: string, limit = 50): Promise<string[]> {
  const { data } = await graphApiClient.get<string[]>('/labels/search', { params: { workspace, q: query, limit } })
  return data
}

export async function getGraphWorkspaces(): Promise<string[]> {
  const { data } = await graphApiClient.get<string[]>('/workspaces')
  return data
}
