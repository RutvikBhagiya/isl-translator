"use client"

import { useEffect, useRef, useState } from "react"

type Landmark = { x: number; y: number; z: number }

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const handsRef = useRef<any>(null)

  const isTrainingRef = useRef(false)
  const trainingSamplesRef = useRef<Landmark[][]>([])
  const gestureNameRef = useRef("")

  const currentSignRef = useRef<string | null>(null)
  const stableCountRef = useRef(0)
  const lastCommittedRef = useRef<string | null>(null)

  const [prediction, setPrediction] = useState("READY")
  const [gestureName, setGestureName] = useState("")
  const [gestureCount, setGestureCount] = useState(0)
  const [status, setStatus] = useState("")
  const [word, setWord] = useState("")

  useEffect(() => {
    let active = true
    let rafId = 0

    const init = async () => {
      const mp = await import("@mediapipe/hands")
      const Hands = mp.Hands

      const hands = new Hands({
        locateFile: (f) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`,
      })

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.8,
        minTrackingConfidence: 0.8,
      })

      hands.onResults((res: any) => {
        if (!active || !canvasRef.current) return

        const ctx = canvasRef.current.getContext("2d")!
        ctx.clearRect(0, 0, 640, 480)
        ctx.drawImage(res.image, 0, 0, 640, 480)

        if (!res.multiHandLandmarks?.length) return
        const hand = res.multiHandLandmarks[0]

        hand.forEach((p: Landmark) => {
          ctx.beginPath()
          ctx.arc(p.x * 640, p.y * 480, 4, 0, Math.PI * 2)
          ctx.fillStyle = "#22c55e"
          ctx.fill()
        })

        if (isTrainingRef.current) {
          trainingSamplesRef.current.push(hand)
          setPrediction(
            `TRAINING ${trainingSamplesRef.current.length}/30`
          )

          if (trainingSamplesRef.current.length >= 30) {
            finishTraining()
          }
          return
        }

        predict(hand)
      })

      handsRef.current = hands
      startCamera()
    }

    const startCamera = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      videoRef.current!.srcObject = stream
      await videoRef.current!.play()

      const loop = async () => {
        if (!active) return
        await handsRef.current.send({ image: videoRef.current })
        rafId = requestAnimationFrame(loop)
      }
      loop()
    }

    init()
    loadGestures()

    return () => {
      active = false
      cancelAnimationFrame(rafId)
      handsRef.current?.close()
    }
  }, [])

  const predict = async (landmarks: Landmark[]) => {
    try {
      const r = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ landmarks }),
      })

      const d = await r.json()
      if (!d.prediction) {
        setPrediction("—")
        return
      }

      setPrediction(d.prediction)

      if (d.prediction === currentSignRef.current) {
        stableCountRef.current += 1
      } else {
        currentSignRef.current = d.prediction
        stableCountRef.current = 1
      }

      if (
        stableCountRef.current >= 3 &&
        d.prediction !== lastCommittedRef.current
      ) {
        setWord((w) => w + d.prediction)
        lastCommittedRef.current = d.prediction
      }
    } catch {
      setPrediction("BACKEND OFFLINE")
    }
  }

  const finishTraining = async () => {
    isTrainingRef.current = false

    const name = gestureNameRef.current.trim().toUpperCase()
    if (!name) {
      setStatus("❌ INVALID NAME")
      return
    }

    await fetch("http://127.0.0.1:8000/train-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        samples: trainingSamplesRef.current,
      }),
    })

    trainingSamplesRef.current = []
    setGestureName("")
    setPrediction("READY")
    setStatus(`✅ SAVED: ${name}`)

    loadGestures()
  }

  const loadGestures = async () => {
    const r = await fetch("http://127.0.0.1:8000/gestures")
    const d = await r.json()
    setGestureCount(d.count)
  }

  return (
    <main className="min-h-screen bg-black flex flex-col items-center justify-center text-white px-4">

      <h1 className="text-3xl font-bold mb-1">ISL Translator</h1>

      <div className="mb-4 text-center">
        <div className="text-xs tracking-[0.4em] text-green-400">
          LIVE WORD
        </div>
        <div className="
          mt-2 px-6 py-4 min-h-[64px]
          rounded-2xl
          bg-green-500/10
          border border-green-500/30
          shadow-[0_0_40px_rgba(34,197,94,0.25)]
        ">
          <span className="text-4xl font-bold tracking-widest text-green-300">
            {word || "—"}
          </span>
        </div>
      </div>

      <p className="text-green-400 text-xl">{prediction}</p>
      <p className="text-sm text-gray-400 mb-3">
        READY ({gestureCount} SIGNS)
      </p>

      <video ref={videoRef} className="hidden" />
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        className="
          rounded-2xl
          border border-green-500/30
          shadow-[0_0_60px_rgba(34,197,94,0.25)]
        "
      />

      <input
        value={gestureName}
        onChange={(e) => {
          setGestureName(e.target.value)
          gestureNameRef.current = e.target.value
        }}
        placeholder="Gesture name (A, B, HELLO)"
        className="
          mt-5 w-full max-w-md
          px-4 py-3
          text-lg text-white
          bg-gray-900
          border border-gray-700
          rounded-lg
          focus:outline-none focus:border-green-500
        "
      />

      <button
        onClick={() => {
          if (!gestureName.trim()) {
            setStatus("❌ ENTER GESTURE NAME")
            return
          }
          trainingSamplesRef.current = []
          isTrainingRef.current = true
          setStatus("🎥 CAPTURING...")
        }}
        className="
          mt-3 bg-green-600 hover:bg-green-700
          px-6 py-3 rounded-lg
          text-lg font-semibold
        "
      >
        Capture Gesture
      </button>

      {status && (
        <p className="mt-3 text-sm text-yellow-400">{status}</p>
      )}
    </main>
  )
}
