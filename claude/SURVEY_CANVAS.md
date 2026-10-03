# サーベイ要約を Slack キャンバスへ書き込む

`run_survey.sh` が PR 作成後に `claude -p` へ渡すプロンプト。末尾に「入力」（キャンバス ID・実行時刻・PR の URL / タイトル / 本文）がスクリプトから追記される。
使ってよいツールは `slack_read_canvas` と `slack_update_canvas` だけ。ファイル操作・Web 検索・メッセージ送信はしない。

## 手順

1. `slack_read_canvas` でキャンバスを読み、`section_id_mapping` から先頭のタイトル見出し（`# デイリーサーベイ`）の section_id を取る。
2. `slack_update_canvas` で、タイトル見出しの section に `edit_type: "append"` で 1 エントリを追加する（最新が一番上に来る）。既存エントリは変更・削除しない。
3. 終わったら「done」とだけ出力する。

## エントリの形式

```
## ![](slack_date:YYYY-MM-DD) HH:MM — <探索軸>: <テーマを短く>

- 読んだもの: <論文・資料名（venue/年）を 2〜4 件>
- 示唆: <KEKG（KG構造→内部表現→編集可塑性の IRT 測定）への示唆を 1〜2 文>
- <必要なら補足 1 行>
- 次の候補: <handoff や TODO から 1〜2 件>
- PR: [knowledge_base_lab#<番号>](<PR URL>)
```

- 日本語で 5〜8 行。PR 本文に書かれていないことは書かない（推測で補わない）。
- 見出しは `##` まで。リスト項目の中に見出し・コードブロック・表を入れない。
- 日付は `![](slack_date:YYYY-MM-DD)` の形だけを使い、曜日などを付けない。時刻は入力の実行時刻（JST）を使う。
