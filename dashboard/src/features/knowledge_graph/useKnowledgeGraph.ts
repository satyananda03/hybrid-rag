import { useCallback, useEffect } from 'react'
import { toast } from 'sonner'
import useSWR from 'swr'
import { fetchKnowledgeGraph, graphCacheKey } from './api'
import { graphFromResponse, mergeRawGraphs, sigmaGraphFromRaw } from './graph'
import { useKnowledgeGraphStore } from './store'
import type { GraphWorkspace } from './types'

const MAX_RETRIES = 3
const RETRY_BASE_DELAY_MS = 1_000

function messageFrom(error: unknown): string {
  return error instanceof Error ? error.message : 'Unable to load knowledge graph.'
}

/**
 * Muat subgraph untuk pengaturan yang sedang aktif.
 *
 * Cache-nya milik SWR: pindah halaman lalu kembali dalam satu menit tidak
 * memanggil `/graphs` sekali pun, dan dua komponen yang meminta subgraph sama
 * berbagi satu request. Percobaan ulang dan pembuangan balasan basi juga sudah
 * dipegang SWR — dulu keduanya ditulis tangan di sini.
 *
 * Grafnya sendiri tetap tinggal di store zustand, bukan di state hook: viewer,
 * panel properti, dan pencarian membacanya dari sana, dan `expandNode` menambal
 * isinya di tempat.
 */
export function useKnowledgeGraph(workspace: GraphWorkspace) {
  const { label, maxDepth, maxNodes } = useKnowledgeGraphStore((state) => state.settings)
  const version = useKnowledgeGraphStore((state) => state.graphVersion)
  const rawGraph = useKnowledgeGraphStore((state) => state.rawGraph)

  const { data, error, isValidating, mutate } = useSWR(
    graphCacheKey({ label, maxDepth, maxNodes, workspace }, version),
    () => fetchKnowledgeGraph({ label, maxDepth, maxNodes, workspace }),
    {
      errorRetryCount: MAX_RETRIES,
      errorRetryInterval: RETRY_BASE_DELAY_MS,
      // Hanya saat benar-benar menembak server, jadi membuka ulang halaman dari
      // cache tidak memunculkan toast yang sama berulang kali.
      onSuccess: (response) => {
        if (response.is_truncated) toast.info(`Graph was limited to ${maxNodes} nodes.`)
      },
      onError: () => toast.error('Unable to load knowledge graph. Refresh to retry.')
    }
  )

  // Store adalah sumber render, jadi penulisannya harus ikut terjadi saat data
  // datang dari cache — bukan hanya saat fetch berhasil.
  useEffect(() => {
    if (!data) return
    const raw = graphFromResponse(data)
    const { typeColors } = sigmaGraphFromRaw(raw)
    useKnowledgeGraphStore.getState().setRawGraph(raw, typeColors)
  }, [data])

  useEffect(() => {
    useKnowledgeGraphStore.getState().setLoading(isValidating)
  }, [isValidating])

  useEffect(() => {
    useKnowledgeGraphStore.getState().setError(error ? messageFrom(error) : null)
  }, [error])

  const expandNode = useCallback(async (nodeId: string) => {
    const store = useKnowledgeGraphStore.getState()
    const node = store.rawGraph?.nodes.find((item) => item.id === nodeId)
    if (!node) return
    store.setLoading(true)
    try {
      // Sengaja lewat di luar cache SWR: parameternya sekali pakai (satu node
      // yang baru saja diklik), jadi menyimpannya hanya mengisi cache dengan
      // entri yang tidak akan pernah diminta lagi.
      const response = await fetchKnowledgeGraph({
        label: node.labels[0] || node.id,
        maxDepth: Math.min(maxDepth, 2), maxNodes, workspace,
      })
      const current = useKnowledgeGraphStore.getState().rawGraph
      if (!current) return
      const merged = mergeRawGraphs(current, graphFromResponse(response))
      const { typeColors } = sigmaGraphFromRaw(merged)
      useKnowledgeGraphStore.getState().setRawGraph(merged, typeColors)
    } catch (cause) {
      toast.error(messageFrom(cause))
    } finally {
      useKnowledgeGraphStore.getState().setLoading(false)
    }
  }, [maxDepth, maxNodes, workspace])

  return { rawGraph, load: () => { void mutate() }, expandNode }
}
