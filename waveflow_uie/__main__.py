"""Entry point for python -m waveflow_uie."""

print("""WaveFlow-UIE: Wavelet-Domain Rectified Flow for Underwater Image Enhancement

Usage:
  python -m waveflow_uie.train    --config configs/waveflow_uie_uieb.yaml
  python -m waveflow_uie.sample   --checkpoint <path> --input <image> --steps 5
  python -m waveflow_uie.evaluate --checkpoint <path> --dataset uieb_test --steps 1,2,5,10,20
  python -m waveflow_uie.baseline_wfdiff --weights pretrained/net_g_405000_UIEB.pth
""")
