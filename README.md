# Semi-PIC
中心密度をディリクレ条件で固定し電子はBoltzmann分布で計算。ただし、電子／イオン共にPIC法で計算されており、電子PICは電子エネルギーの計算のためだけに存在している

# 固定中心密度を使う場合
USE_BOLTZMANN_ELECTRON=1, NE_CENTER_MODE=1, NE_CENTER_FIXED=1e16

# 時間依存中心密度を使う場合
USE_BOLTZMANN_ELECTRON=1, NE_CENTER_MODE=2, NE_CENTER_TABLE=Time-dependent-ne.csv

# 電子密度のテーブルは線形補間される

# 電子温度は計算されたEEPFから引用される
