"use client"

import { useEffect, useRef, useState } from "react"

let lastSent = 0
let lastSpoken = ""

type Landmark = { x: number; y: number; z: number }
type HandLandmarks = Landmark[]

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const lastLandmarksRef = useRef<HandLandmarks | null>(null)

  const [prediction, setPrediction] = useState("")

  const saveGesture = async () => {
    const landmarks = lastLandmarksRef.current
    if (!landmarks) {
      alert("No hand detected")
      return
    }

    const name = prompt("Enter gesture name")
    if (!name) return

    await fetch("http://127.0.0.1:8000/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, landmarks }),
    })

    alert("Saved: " + name)
  }

  useEffect(() => {
    if (!videoRef.current || !canvasRef.current) return
    if (typeof window === "undefined") return

    let camera: any
    let hands: any
    let mounted = true

    const init = async () => {

      const handsModule = await import("@mediapipe/hands")
      const cameraModule = await import("@mediapipe/camera_utils")

      const Hands = handsModule.Hands
      const Camera = cameraModule.Camera

      hands = new Hands({
        locateFile: (file: string) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      })

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7,
      })

      hands.onResults((results: any) => {
        if (!mounted) return

        const video = videoRef.current!
        const canvas = canvasRef.current!
        const ctx = canvas.getContext("2d")!

        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

        if (!results.multiHandLandmarks?.length) return

        const hand: HandLandmarks = results.multiHandLandmarks[0]
        lastLandmarksRef.current = hand

        hand.forEach((p) => {
          ctx.beginPath()
          ctx.arc(
            p.x * canvas.width,
            p.y * canvas.height,
            5,
            0,
            Math.PI * 2
          )
          ctx.fillStyle = "lime"
          ctx.fill()
        })

        const now = Date.now()
        if (now - lastSent < 700) return
        lastSent = now

        fetch("http://127.0.0.1:8000/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ landmarks: hand }),
        })
          .then((res) => res.json())
          .then((data) => {
            if (!mounted) return

            setPrediction(data.gesture)

            if (
              data.gesture &&
              data.gesture !== "UNKNOWN" &&
              data.gesture !== lastSpoken
            ) {
              lastSpoken = data.gesture
              speechSynthesis.speak(
                new SpeechSynthesisUtterance(data.gesture)
              )
            }
          })
          .catch(console.error)
      })

      camera = new Camera(videoRef.current!, {
        width: 640,
        height: 480,
        onFrame: async () => {
          await hands.send({ image: videoRef.current! })
        },
      })

      camera.start()
    }

    init()

    return () => {
      mounted = false
      camera?.stop()
      hands?.close()
    }
  }, [])

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-black">
      <h1 className="text-white text-2xl mb-2">
        ISL Instant Translator
      </h1>

      <h2 className="text-green-400 text-xl mb-2">
        {prediction}
      </h2>

      <video ref={videoRef} className="hidden" />

      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        className="rounded-xl border border-gray-500"
      />

      <button
        onClick={saveGesture}
        className="mt-4 bg-green-500 px-4 py-2 rounded"
      >
        Save Gesture
      </button>
    </main>
  )
}
