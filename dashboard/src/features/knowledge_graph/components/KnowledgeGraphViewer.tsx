import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { SigmaContainer, useRegisterEvents, useSetSettings, useSigma } from '@react-sigma/core'
import { useLayoutCirclepack } from '@react-sigma/layout-circlepack'
import { EdgeArrowProgram, EdgeLineProgram, EdgeRectangleProgram, NodeCircleProgram, NodePointProgram } from 'sigma/rendering'
import { NodeBorderProgram } from '@sigma/node-border'
import { EdgeCurvedArrowProgram, createEdgeCurveProgram } from '@sigma/edge-curve'
import forceAtlas2 from 'graphology-layout-forceatlas2'
import ForceAtlas2Supervisor from 'graphology-layout-forceatlas2/worker'
import ForceSupervisor from 'graphology-layout-force/worker'
import NoverlapSupervisor from 'graphology-layout-noverlap/worker'
import type { MultiUndirectedGraph } from 'graphology'
import { GraphControls } from '@/components/knowledge_graph/GraphControls'
import { GraphLabelSearch } from '@/components/knowledge_graph/GraphLabelSearch'
import { GraphLegend } from '@/components/knowledge_graph/GraphLegend'
import { GraphProperties } from '@/components/knowledge_graph/GraphProperties'
import { GraphSearch } from '@/components/knowledge_graph/GraphSearch'
import { assignCircularLayout, assignRandomLayout, sigmaGraphFromRaw } from '../graph'
import { useKnowledgeGraphStore, type EdgeDisplayData, type GraphLayout, type NodeDisplayData } from '../store'
import { useKnowledgeGraph } from '../useKnowledgeGraph'
import type { GraphWorkspace } from '../types'
import '@react-sigma/core/lib/style.css'

const EDGE_PERFORMANCE_LIMIT = 5_000
const LABEL_PERFORMANCE_LIMIT = 2_000
const workerBudget = (order: number) => Math.min(1_500 + order / 10, 10_000)

// Ported from graph_ui/src/lib/constants.ts so the dark-theme palette matches
// localhost:8000/webui exactly (this dashboard is dark-mode only).
const LABEL_COLOR_DARK = '#FFFFFF'
const LABEL_COLOR_HIGHLIGHTED_DARK = '#000000'
const NODE_COLOR_DISABLED = '#E2E2E2'
const NODE_BORDER_COLOR_SELECTED = '#F57F17'
const EDGE_COLOR_DARK = '#888888'
const EDGE_COLOR_SELECTED = '#F57F17'
const EDGE_COLOR_HIGHLIGHTED_DARK = '#F57F17'

type Graph = MultiUndirectedGraph<NodeDisplayData, EdgeDisplayData>
type LayoutSupervisor = { start: () => void; stop: () => void; kill: () => void }

function sigmaSettings(enableEdgeEvents: boolean, useCurvedEdges: boolean) {
  return {
    allowInvalidContainer: true,
    defaultNodeType: 'border',
    defaultEdgeType: useCurvedEdges ? 'curvedNoArrow' : 'rect',
    defaultEdgeColor: EDGE_COLOR_DARK,
    hideEdgesOnMove: true,
    enableEdgeEvents,
    labelRenderedSizeThreshold: 12,
    labelGridCellSize: 60,
    labelSize: 12,
    edgeLabelSize: 8,
    renderEdgeLabels: false,
    labelColor: { color: LABEL_COLOR_DARK, attribute: 'labelColor' },
    edgeLabelColor: { color: LABEL_COLOR_DARK, attribute: 'labelColor' },
    edgeProgramClasses: {
      rect: EdgeRectangleProgram,
      line: EdgeLineProgram,
      arrow: EdgeArrowProgram,
      curvedArrow: EdgeCurvedArrowProgram,
      curvedNoArrow: createEdgeCurveProgram(),
    },
    nodeProgramClasses: {
      point: NodePointProgram,
      default: NodePointProgram,
      circle: NodeCircleProgram,
      border: NodeBorderProgram,
    },
  }
}

export function KnowledgeGraphViewer({ workspace }: { workspace: GraphWorkspace }) {
  const dataVersion = useKnowledgeGraphStore((state) => state.dataVersion)
  const edgeCount = useKnowledgeGraphStore((state) => state.edgeCount)
  const isLoading = useKnowledgeGraphStore((state) => state.isLoading)
  const isLayoutRunning = useKnowledgeGraphStore((state) => state.isLayoutRunning)
  const error = useKnowledgeGraphStore((state) => state.error)
  const minEdgeSize = useKnowledgeGraphStore((state) => state.settings.minEdgeSize)
  const maxEdgeSize = useKnowledgeGraphStore((state) => state.settings.maxEdgeSize)
  const { expandNode, load } = useKnowledgeGraph(workspace)
  const edgeEvents = useKnowledgeGraphStore((state) => state.settings.enableEdgeEvents) && edgeCount <= EDGE_PERFORMANCE_LIMIT
  const useCurvedEdges = edgeCount > 0 && edgeCount <= EDGE_PERFORMANCE_LIMIT
  // SigmaContainer rebuilds its renderer whenever its settings reference
  // changes. Keep this object stable unless its two construction-time flags
  // actually change; `createEdgeCurveProgram()` otherwise creates a new
  // program class on every page/store render.
  const rendererSettings = useMemo(
    () => sigmaSettings(edgeEvents, useCurvedEdges),
    [edgeEvents, useCurvedEdges],
  )

  // Build a fresh graphology instance whenever the underlying data changes
  // (fetch, expand, prune, edit, rename) or the edge-size range setting
  // changes. This is intentionally NOT mutated in place: Sigma keeps
  // internal render caches per bound graph instance, and clearing/
  // repopulating a live-bound instance corrupts them (surfaces as "edge
  // can't be repaint"). A fresh instance + remounting SigmaContainer (via
  // `key`) below is the same pattern the reference graph_ui app uses.
  //
  // Keyed on `dataVersion`, NOT on the `rawGraph` reference: node drag
  // rewrites `rawGraph` (to persist x/y) without bumping dataVersion, and
  // rebuilding here would hand SigmaContainer a new instance -- it recreates
  // its whole Sigma renderer on any `graph` identity change -- resetting every
  // node that has no persisted x/y back to its hash position.
  const graph = useMemo(
    () => sigmaGraphFromRaw(
      useKnowledgeGraphStore.getState().rawGraph ?? { nodes: [], edges: [], isTruncated: false },
      { min: minEdgeSize, max: maxEdgeSize },
    ).graph,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [dataVersion, minEdgeSize, maxEdgeSize],
  )

  return (
    <div className="relative h-full w-full overflow-hidden bg-surface-container-low">
      <SigmaContainer
        key={`${dataVersion}-${edgeEvents}-${useCurvedEdges}-${minEdgeSize}-${maxEdgeSize}`}
        graph={graph}
        settings={rendererSettings}
        className="!size-full !bg-surface-container-low"
      >
        <GraphCanvas graph={graph} workspace={workspace} onExpand={expandNode} />
      </SigmaContainer>
      {(isLoading || isLayoutRunning) && (
        <LoadingOverlay label={isLoading ? 'Loading graph data…' : 'Computing layout…'} />
      )}
      {error && <ErrorOverlay message={error} onRetry={load} />}
    </div>
  )
}

function GraphCanvas({
  graph,
  workspace,
  onExpand,
}: {
  graph: Graph
  workspace: GraphWorkspace
  onExpand: (nodeId: string) => void
}) {
  const sigma = useSigma<NodeDisplayData, EdgeDisplayData>()
  const registerEvents = useRegisterEvents<NodeDisplayData, EdgeDisplayData>()
  const setSettings = useSetSettings<NodeDisplayData, EdgeDisplayData>()
  const layoutCirclepack = useLayoutCirclepack()
  const settings = useKnowledgeGraphStore((state) => state.settings)
  const selectedNode = useKnowledgeGraphStore((state) => state.selectedNode)
  const selectedEdge = useKnowledgeGraphStore((state) => state.selectedEdge)
  const focusedNode = useKnowledgeGraphStore((state) => state.focusedNode)
  const focusedEdge = useKnowledgeGraphStore((state) => state.focusedEdge)
  const moveToSelectedNode = useKnowledgeGraphStore((state) => state.moveToSelectedNode)
  const edgeCount = useKnowledgeGraphStore((state) => state.edgeCount)
  const selectNode = useKnowledgeGraphStore((state) => state.selectNode)
  const selectEdge = useKnowledgeGraphStore((state) => state.selectEdge)
  const focusNode = useKnowledgeGraphStore((state) => state.focusNode)
  const focusEdge = useKnowledgeGraphStore((state) => state.focusEdge)
  const clearSelection = useKnowledgeGraphStore((state) => state.clearSelection)
  const setLayoutRunning = useKnowledgeGraphStore((state) => state.setLayoutRunning)
  const updateNodePosition = useKnowledgeGraphStore((state) => state.updateNodePosition)
  const supervisor = useRef<LayoutSupervisor | null>(null)
  const layoutTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  useEffect(() => {
    if (typeof document === 'undefined') return
    const sync = () => setIsFullscreen(document.fullscreenElement === sigma.getContainer())
    document.addEventListener('fullscreenchange', sync)
    return () => document.removeEventListener('fullscreenchange', sync)
  }, [sigma])

  const effectiveEdgeEvents = settings.enableEdgeEvents && edgeCount <= EDGE_PERFORMANCE_LIMIT

  const stopLayout = useCallback(() => {
    if (layoutTimer.current) clearTimeout(layoutTimer.current)
    layoutTimer.current = null
    try { supervisor.current?.stop(); supervisor.current?.kill() } catch { /* worker already closed */ }
    supervisor.current = null
    setLayoutRunning(false)
  }, [setLayoutRunning])

  const runLayout = useCallback((layout: GraphLayout) => {
    stopLayout()
    if (graph.order === 0) return
    if (layout === 'circular' || layout === 'random' || layout === 'circlepack') {
      setLayoutRunning(true)
      const frame = typeof requestAnimationFrame === 'undefined' ? setTimeout : requestAnimationFrame
      frame(() => {
        if (layout === 'circular') assignCircularLayout(graph)
        else if (layout === 'random') assignRandomLayout(graph)
        else {
          const positions = layoutCirclepack.positions()
          graph.updateEachNodeAttributes((node, attributes) => {
            const position = positions[node]
            return position ? { ...attributes, ...position } : attributes
          }, { attributes: ['x', 'y'] })
        }
        sigma.setCustomBBox(null)
        sigma.refresh()
        sigma.getCamera().animatedReset()
        setLayoutRunning(false)
      })
      return
    }
    try {
      if (layout === 'forceAtlas2') supervisor.current = new ForceAtlas2Supervisor(graph as never, { settings: forceAtlas2.inferSettings(graph.order) })
      else if (layout === 'force') supervisor.current = new ForceSupervisor(graph as never, { settings: { attraction: 0.0003, repulsion: 0.02, gravity: 0.02, inertia: 0.8, maxMove: 5 } })
      else supervisor.current = new NoverlapSupervisor(graph as never, { settings: { margin: 10, expansion: 1.1, gridSize: 1, ratio: 1, speed: 3 } })
      sigma.setCustomBBox(null)
      supervisor.current.start()
      setLayoutRunning(true)
      layoutTimer.current = setTimeout(() => { stopLayout(); sigma.refresh() }, workerBudget(graph.order))
    } catch {
      stopLayout()
    }
  }, [graph, layoutCirclepack, setLayoutRunning, sigma, stopLayout])

  const persistDraggedPosition = updateNodePosition

  useEffect(() => {
    // SigmaContainer is remounted (via `key`) whenever the data changes, so
    // this component mounts fresh with `graph` already fully built and bound.
    // Running the initial layout once per mount is sufficient.
    if (graph.order > 1) runLayout('forceAtlas2')
    return stopLayout
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!moveToSelectedNode) return
    if (selectedNode && graph.hasNode(selectedNode)) {
      const position = graph.getNodeAttributes(selectedNode)
      sigma.getCamera().animate({ x: position.x, y: position.y, ratio: 0.45 }, { duration: 250 })
    }
    // A camera move is an explicit, one-shot Find action. Normal node clicks
    // select the entity for the property panel but keep the current view.
    selectNode(selectedNode)
  }, [graph, moveToSelectedNode, selectNode, selectedNode, sigma])

  // Reducer logic ported from graph_ui/src/components/graph/GraphControl.tsx
  // so hover/selection highlighting matches localhost:8000/webui exactly:
  // focused node -> node+neighbours highlighted (border amber if selected),
  // everything else greyed to NODE_COLOR_DISABLED; edges either hidden or
  // recolored amber depending on the "hide unselected edges" setting.
  // Focused edge (no node focus) -> endpoints highlighted + shrunk, the edge
  // itself colored amber (selected) or highlighted-amber (hover).
  useEffect(() => {
    const activeNode = focusedNode ?? selectedNode
    const activeEdge = effectiveEdgeEvents ? focusedEdge ?? selectedEdge : null
    const nodeFocusValid = Boolean(activeNode && graph.hasNode(activeNode))
    const edgeFocusValid = !nodeFocusValid && Boolean(activeEdge && graph.hasEdge(activeEdge))

    const neighbours = new Set<string>()
    if (nodeFocusValid && activeNode) graph.forEachNeighbor(activeNode, (node) => neighbours.add(node))
    let edgeSource = ''
    let edgeTarget = ''
    if (edgeFocusValid && activeEdge) {
      edgeSource = graph.source(activeEdge)
      edgeTarget = graph.target(activeEdge)
    }

    const focusNodeReducer = nodeFocusValid || edgeFocusValid ? (node: string, data: NodeDisplayData) => {
      const next = { ...data, highlighted: false, labelColor: LABEL_COLOR_DARK }
      if (nodeFocusValid) {
        if (node === activeNode || neighbours.has(node)) {
          next.highlighted = true
          next.labelColor = LABEL_COLOR_HIGHLIGHTED_DARK
          if (node === selectedNode) next.borderColor = NODE_BORDER_COLOR_SELECTED
        } else {
          next.color = NODE_COLOR_DISABLED
        }
      } else if (edgeFocusValid) {
        if (node === edgeSource || node === edgeTarget) {
          next.highlighted = true
          next.size = 3
          next.labelColor = LABEL_COLOR_HIGHLIGHTED_DARK
        } else {
          next.color = NODE_COLOR_DISABLED
        }
      }
      return next
    } : null
    // A hidden entity type wins over focus styling; sigma also drops its edges.
    const hiddenTypes = new Set(settings.hiddenEntityTypes)

    setSettings({
      enableEdgeEvents: effectiveEdgeEvents,
      renderLabels: settings.showLabels && graph.order <= LABEL_PERFORMANCE_LIMIT,
      renderEdgeLabels: settings.showEdgeLabels && graph.size <= EDGE_PERFORMANCE_LIMIT,
      nodeReducer: hiddenTypes.size > 0 || focusNodeReducer ? (node, data) => {
        if (hiddenTypes.has(data.entityType)) return { ...data, hidden: true }
        return focusNodeReducer ? focusNodeReducer(node, data) : data
      } : null,
      edgeReducer: nodeFocusValid || edgeFocusValid ? (edge, data) => {
        const next = { ...data, hidden: false, labelColor: LABEL_COLOR_DARK, color: EDGE_COLOR_DARK }
        if (nodeFocusValid) {
          const touchesActive = graph.source(edge) === activeNode || graph.target(edge) === activeNode
          if (settings.hideUnselectedEdges) {
            if (!touchesActive) next.hidden = true
          } else if (touchesActive) {
            next.color = EDGE_COLOR_HIGHLIGHTED_DARK
          }
        } else if (edgeFocusValid) {
          if (edge === selectedEdge) next.color = EDGE_COLOR_SELECTED
          else if (edge === activeEdge) next.color = EDGE_COLOR_HIGHLIGHTED_DARK
          else if (settings.hideUnselectedEdges) next.hidden = true
        }
        return next
      } : null,
    })
  }, [effectiveEdgeEvents, focusedEdge, focusedNode, graph, selectedEdge, selectedNode, setSettings, settings.hiddenEntityTypes, settings.hideUnselectedEdges, settings.showEdgeLabels, settings.showLabels])

  // Edge sizes are baked in at graph-build time (see sigmaGraphFromRaw), so
  // no separate effect mutates edge attributes here — mutating edges on a
  // live-bound Sigma graph is what caused the "can't be repaint" crash.

  useEffect(() => {
    registerEvents({
      enterNode: ({ node }) => focusNode(node), leaveNode: () => focusNode(null),
      clickNode: ({ node }) => selectNode(node), clickStage: () => clearSelection(),
      ...(effectiveEdgeEvents ? {
        enterEdge: ({ edge }) => focusEdge(graph.getEdgeAttribute(edge, 'rawId')),
        leaveEdge: () => focusEdge(null),
        clickEdge: ({ edge }) => selectEdge(graph.getEdgeAttribute(edge, 'rawId')),
      } : {}),
    })
  }, [clearSelection, effectiveEdgeEvents, focusEdge, focusNode, graph, registerEvents, selectEdge, selectNode])

  return <><GraphDrag enabled={settings.enableDrag} onPosition={persistDraggedPosition} /><div className="absolute top-3 left-3 z-20 flex items-start gap-2"><GraphLabelSearch workspace={workspace} />{settings.showSearch && <GraphSearch />}</div><GraphControls onLayout={runLayout} onZoom={(direction) => { const camera = sigma.getCamera(); if (direction === 'reset') { sigma.setCustomBBox(null); camera.animatedReset() } else camera.animate({ ratio: camera.ratio * (direction === 'in' ? 0.65 : 1.55) }, { duration: 180 }) }} isFullscreen={isFullscreen} onFullscreen={() => toggleFullscreen(sigma.getContainer(), setIsFullscreen)} />{settings.showLegend && <div className="absolute right-3 bottom-3 z-20"><GraphLegend /></div>}{settings.showProperties && <GraphProperties onExpand={onExpand} />}<div className="absolute bottom-3 left-16 text-xs text-on-surface-variant">Depth: {settings.maxDepth} · Nodes: {graph.order} · Edges: {graph.size}</div></>
}

function GraphDrag({ enabled, onPosition }: { enabled: boolean; onPosition: (nodeId: string, position: { x: number; y: number }) => void }) {
  const sigma = useSigma<NodeDisplayData, EdgeDisplayData>()
  const registerEvents = useRegisterEvents<NodeDisplayData, EdgeDisplayData>()
  const [draggedNode, setDraggedNode] = useState<string | null>(null)
  useEffect(() => {
    if (!enabled) return
    registerEvents({
      downNode: ({ node }) => { setDraggedNode(node); sigma.getGraph().setNodeAttribute(node, 'highlighted', true) },
      mousemovebody: (event) => {
        if (!draggedNode) return
        const position = sigma.viewportToGraph(event)
        sigma.getGraph().mergeNodeAttributes(draggedNode, position)
        event.preventSigmaDefault(); event.original.preventDefault(); event.original.stopPropagation()
      },
      mouseup: () => {
        if (!draggedNode) return
        const position = sigma.getGraph().getNodeAttributes(draggedNode)
        sigma.getGraph().removeNodeAttribute(draggedNode, 'highlighted')
        onPosition(draggedNode, position)
        setDraggedNode(null)
      },
      mousedown: ({ original }) => {
        if ((original as MouseEvent).buttons !== 0 && !sigma.getCustomBBox()) sigma.setCustomBBox(sigma.getBBox())
      },
    })
  }, [draggedNode, enabled, onPosition, registerEvents, sigma])
  return null
}

function toggleFullscreen(container: HTMLElement, setFullscreen: (value: boolean) => void) {
  if (typeof document === 'undefined') return
  if (document.fullscreenElement) {
    void document.exitFullscreen().then(() => setFullscreen(false)).catch(() => undefined)
    return
  }
  if (typeof container.requestFullscreen !== 'function') return
  void container.requestFullscreen().then(() => setFullscreen(true)).catch(() => undefined)
}

function LoadingOverlay({ label }: { label: string }) {
  return <div className="absolute inset-0 z-30 grid place-items-center bg-surface-container-low/80 backdrop-blur-sm"><div className="text-center text-sm text-on-surface"><span className="mx-auto mb-2 block size-7 animate-spin rounded-full border-4 border-primary border-t-transparent" />{label}</div></div>
}

function ErrorOverlay({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="absolute inset-0 z-40 grid place-items-center bg-surface-container-low/90 p-6 text-center backdrop-blur-sm">
      <div className="max-w-sm space-y-3">
        <p className="text-sm text-on-surface">{message}</p>
        <button
          type="button"
          onClick={onRetry}
          className="rounded bg-primary px-3 py-1.5 text-sm font-semibold text-on-primary"
        >
          Retry
        </button>
      </div>
    </div>
  )
}
