import MiniSearch from 'minisearch'
import { create } from 'zustand'
import type { Attributes } from 'graphology-types'
import type { KnowledgeGraphEdge, KnowledgeGraphNode } from './types'

export interface RawNode extends KnowledgeGraphNode {
  degree: number
  x?: number
  y?: number
}

export interface RawEdge extends KnowledgeGraphEdge {
  dynamicId?: string
}

export interface RawGraph {
  nodes: RawNode[]
  edges: RawEdge[]
  isTruncated: boolean
}

export interface NodeDisplayData extends Attributes {
  x: number
  y: number
  label: string
  color: string
  labelColor?: string
  size: number
  borderColor: string
  highlighted?: boolean
}

export interface EdgeDisplayData extends Attributes {
  rawId: string
  label?: string
  originalWeight: number
  size: number
  hidden?: boolean
  color?: string
}

export type GraphLayout = 'circular' | 'circlepack' | 'random' | 'forceAtlas2' | 'force' | 'noverlap'

export interface GraphSettings {
  label: string
  maxDepth: number
  maxNodes: number
  showLabels: boolean
  showEdgeLabels: boolean
  showLegend: boolean
  showProperties: boolean
  showSearch: boolean
  enableDrag: boolean
  hideUnselectedEdges: boolean
  enableEdgeEvents: boolean
  minEdgeSize: number
  maxEdgeSize: number
}

interface KnowledgeGraphState {
  rawGraph: RawGraph | null
  selectedNode: string | null
  selectedEdge: string | null
  focusedNode: string | null
  focusedEdge: string | null
  /** True only when an explicit Find result should move the camera. */
  moveToSelectedNode: boolean
  typeColors: Map<string, string>
  searchIndex: MiniSearch<{ id: string; label: string }> | null
  nodeCount: number
  edgeCount: number
  isLoading: boolean
  isLayoutRunning: boolean
  error: string | null
  graphVersion: number
  /**
   * Bumped whenever `rawGraph` content changes in a way that requires
   * rebuilding the Sigma-bound graphology instance from scratch (fetch,
   * expand, prune, rename, property edits). Node drag does NOT bump this —
   * dragging only updates x/y on the existing instance without a rebuild.
   * Consumers use this to remount SigmaContainer instead of mutating a
   * live-bound graph instance in place.
   */
  dataVersion: number
  settings: GraphSettings
  setRawGraph: (graph: RawGraph | null, colors?: Map<string, string>) => void
  setLoading: (loading: boolean) => void
  setLayoutRunning: (running: boolean) => void
  setError: (error: string | null) => void
  selectNode: (id: string | null, moveTo?: boolean) => void
  selectEdge: (id: string | null) => void
  focusNode: (id: string | null) => void
  focusEdge: (id: string | null) => void
  clearSelection: () => void
  updateSettings: (settings: Partial<GraphSettings>) => void
  refresh: () => void
  updateNodePosition: (nodeId: string, position: { x: number; y: number }) => void
  pruneNode: (nodeId: string) => void
}

const defaultSettings: GraphSettings = {
  label: '*', maxDepth: 3, maxNodes: 3000, showLabels: true, showEdgeLabels: false,
  showLegend: false, showProperties: true, showSearch: true, enableDrag: true,
  hideUnselectedEdges: true, enableEdgeEvents: false, minEdgeSize: 1, maxEdgeSize: 5,
}

const createSearchIndex = (rawGraph: RawGraph | null) => {
  if (!rawGraph || rawGraph.nodes.length > 50_000) return null
  const index = new MiniSearch<{ id: string; label: string }>({
    idField: 'id', fields: ['label'], searchOptions: { prefix: true, fuzzy: 0.2 },
  })
  index.addAll(rawGraph.nodes.map((node) => ({ id: node.id, label: `${node.id} ${node.labels.join(' ')}` })))
  return index
}

const patchGraph = (
  state: KnowledgeGraphState,
  patch: (graph: RawGraph) => RawGraph,
): Pick<KnowledgeGraphState, 'rawGraph' | 'searchIndex' | 'nodeCount' | 'edgeCount'> => {
  const rawGraph = state.rawGraph ? patch(state.rawGraph) : null
  return {
    rawGraph,
    searchIndex: createSearchIndex(rawGraph),
    nodeCount: rawGraph?.nodes.length ?? 0,
    edgeCount: rawGraph?.edges.length ?? 0,
  }
}

export const useKnowledgeGraphStore = create<KnowledgeGraphState>((set) => ({
  rawGraph: null,
  selectedNode: null,
  selectedEdge: null,
  focusedNode: null,
  focusedEdge: null,
  moveToSelectedNode: false,
  typeColors: new Map(),
  searchIndex: null,
  nodeCount: 0,
  edgeCount: 0,
  isLoading: false,
  isLayoutRunning: false,
  error: null,
  graphVersion: 0,
  dataVersion: 0,
  settings: defaultSettings,
  setRawGraph: (rawGraph, typeColors = new Map()) => set((state) => ({
    rawGraph, typeColors, searchIndex: createSearchIndex(rawGraph),
    nodeCount: rawGraph?.nodes.length ?? 0, edgeCount: rawGraph?.edges.length ?? 0,
    selectedNode: null, selectedEdge: null, focusedNode: null, focusedEdge: null,
    moveToSelectedNode: false,
    dataVersion: state.dataVersion + 1,
  })),
  setLoading: (isLoading) => set({ isLoading }),
  setLayoutRunning: (isLayoutRunning) => set({ isLayoutRunning }),
  setError: (error) => set({ error }),
  selectNode: (selectedNode, moveToSelectedNode = false) => set({ selectedNode, selectedEdge: null, moveToSelectedNode }),
  selectEdge: (selectedEdge) => set({ selectedEdge, selectedNode: null, moveToSelectedNode: false }),
  focusNode: (focusedNode) => set({ focusedNode }),
  focusEdge: (focusedEdge) => set({ focusedEdge }),
  clearSelection: () => set({ selectedNode: null, selectedEdge: null, focusedNode: null, focusedEdge: null, moveToSelectedNode: false }),
  updateSettings: (settings) => set((state) => ({ settings: { ...state.settings, ...settings } })),
  refresh: () => set((state) => ({ graphVersion: state.graphVersion + 1 })),
  // Node drag: intentionally does NOT bump dataVersion. This only updates
  // x/y on the raw graph so a future rebuild keeps the dragged position; the
  // live Sigma-bound instance's coordinates are already updated directly by
  // the drag handler for immediate visual feedback, with no remount needed.
  updateNodePosition: (nodeId, position) => set((state) => patchGraph(state, (graph) => ({
    ...graph,
    nodes: graph.nodes.map((node) => node.id === nodeId ? { ...node, ...position } : node),
  }))),
  pruneNode: (nodeId) => set((state) => ({
    ...patchGraph(state, (graph) => ({
      ...graph,
      nodes: graph.nodes.filter((node) => node.id !== nodeId),
      edges: graph.edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId),
    })),
    selectedNode: state.selectedNode === nodeId ? null : state.selectedNode,
    focusedNode: state.focusedNode === nodeId ? null : state.focusedNode,
    dataVersion: state.dataVersion + 1,
  })),
}))

export function selectedElement(
  rawGraph: RawGraph | null,
  nodeId: string | null,
  edgeId: string | null,
): { type: 'node'; element: RawNode } | { type: 'edge'; element: RawEdge } | null {
  if (!rawGraph) return null
  if (nodeId) {
    const element = rawGraph.nodes.find((node) => node.id === nodeId)
    return element ? { type: 'node', element } : null
  }
  if (edgeId) {
    const element = rawGraph.edges.find((edge) => edge.id === edgeId)
    return element ? { type: 'edge', element } : null
  }
  return null
}
