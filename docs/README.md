# ドキュメント案内

研究の判断・実装・評価の基準として使う **baseline** と、背景・探索的仮説・過去の提案・対話原本を収める **reference** に分けています。形式（Markdown / PDF）ではなく、現在の研究での役割で分類します。

## ベースライン

### 研究対象と判断基準

- [research_philosophy.md](baseline/research_philosophy.md): 研究の中心命題、増築・環境先行・棄却の方法論
- [intelligent_subject.md](baseline/intelligent_subject.md): 研究対象「知性主体」の定義と判断軸
- [architecture_principles.md](baseline/architecture_principles.md): h/m/i/a の役割分担と拡張原則
- [environment_first_roadmap.md](baseline/environment_first_roadmap.md): 環境先行の進め方と破綻の分類手順。到達段階の表記は実験 7–8 時点

研究哲学・知性主体は実験の判断に直接使うため baseline に置きます。ただし、そこに含まれる作業仮説は実験で更新する対象です。

### モデルと評価設計

- [model_spec.md](baseline/model_spec.md): 採餌モデルの基本構造。現行の感覚適応・energy・内受容を含む仕様は [ルート README](../README.md#最小モデル現行) と次の再設計文書を参照
- [redesign_viability_first.md](baseline/redesign_viability_first.md): viability-first の実装・評価設計。実施済み段階の結果は以下を参照

### 比較の基準となる実験記録

- [summary_viability_arc.md](baseline/summary_viability_arc.md): 実験 9–12b の統合結果と主張の限界。現在の結論を読む入口
- [experiment_log.md](baseline/experiment_log.md): 初期実験から実験 12b までの逐次記録
- [experiment_odor_field.md](baseline/experiment_odor_field.md): 実験 9 の匂い場・感覚適応の詳細

実験記録は過去の結果も含めて比較・検証に使うため baseline に残します。記録中の暫定結論や「次のステップ」は、その実験時点の記述です。

## 参考資料

### 背景・概念整理・探索的な拡張

- [model_inadequacy_research_position.md](reference/model_inadequacy_research_position.md): 追加PDFと現行研究の接続。適応不全の検出・生涯内更新への展望、既存方針との分岐、未検証の隔たり
- [concept_lattice_role_locus.md](reference/concept_lattice_role_locus.md): 役割・所在・時間スケールの概念整理と文献マップ
- [cognition_coupling_principles.md](reference/cognition_coupling_principles.md): 未実装・長期探索段階の認知（LLM）との結合原則
- [discussion_summary.md](reference/discussion_summary.md): 初期議論の要約
- [genealogy_central_hypothesis.md](reference/genealogy_central_hypothesis.md): 中心仮説に至った系譜

### 過去のレビュー・計画・提案

- [critical_review_meta_neuro_evo.md](reference/critical_review_meta_neuro_evo.md): 実験 0–9 を対象とした批判的レビュー。冒頭に後続実験への更新メモあり
- [poc_plan.md](reference/poc_plan.md): 初期 PoC 計画
- [experiment_plan.md](reference/experiment_plan.md): 初期 PoC・採餌モデルの実験計画。現行の評価設計は viability-first 文書へ
- [analysis_primitive_foraging.md](reference/analysis_primitive_foraging.md): 感覚適応導入前の生物との比較・仮説・増築提案
- [next_step_odor_adaptation.md](reference/next_step_odor_adaptation.md): 感覚適応を次の一手とした当時の提案メモ（旧 etc/）

### 対話原本（PDF）

- [最小の知性とはなにか.pdf](reference/最小の知性とはなにか.pdf): 「今日は何をしようか」から最小エージェントと初期 PoC に至る対話記録（98ページ）。研究の成立経緯を確認する原本
- [生体モデルにしたAI - 予測モデル不足の検出_VEVENT.pdf](<reference/生体モデルにしたAI - 予測モデル不足の検出_VEVENT.pdf>): 局所的な誤差検出、モデル切替、神経回路の発達・選択・恒常性を検討した対話記録（15ページ）。今後の学習・構造変更を考える探索資料

PDF は対話中の説明・仮説・提案を含む原本として reference に置きます。採用済みの実装仕様や実験結果を確認するときは baseline を参照してください。

図は [assets/](assets/) にまとめています。
