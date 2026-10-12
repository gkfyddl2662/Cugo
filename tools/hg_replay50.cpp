// Differential replay of games recorded from another Shin Matgo implementation.
//
// The trace (one line per event) gives the full deck order, every decision of both players,
// a state snapshot at the start of every turn and the final result. Each game is replayed
// through the Cugo engine with the recorded draws forced, and every snapshot and the result
// are compared. Card numbers in the trace are 1-based (Cugo CardId + 1); 52 is the dummy
// card a bomb leaves in hand.
//
// usage: cugo_hg_replay50 <trace.txt> [max_reports]
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "cugo/game/policy50.h"
#include "cugo/game/rollout50.h"

namespace {

using cugo::core::CardId;
using cugo::core::CardMask;
using namespace cugo::game;

std::string field(const std::string& line, const std::string& key) {
  const std::string probe = " " + key + "=";
  std::size_t at = line.find(probe);
  if (at == std::string::npos) {
    if (line.rfind(key + "=", 0) != 0) return "";
    at = 0;
  } else {
    at += 1;
  }
  at += key.size() + 1;
  std::size_t end = at;
  int depth = 0;
  while (end < line.size() && (depth > 0 || line[end] != ' ')) {
    if (line[end] == '[') ++depth;
    if (line[end] == ']') --depth;
    ++end;
  }
  return line.substr(at, end - at);
}

std::vector<int> ints(std::string s) {
  std::vector<int> out;
  for (char& c : s) if (c == '[' || c == ']' || c == ',') c = ' ';
  std::istringstream in(s);
  int v;
  while (in >> v) out.push_back(v);
  return out;
}

long long num(const std::string& line, const std::string& key) {
  const std::string v = field(line, key);
  if (v.empty() || v == "True" || v == "False") return v == "True" ? 1 : 0;
  return std::stoll(v);
}

CardMask mask_of(const std::vector<int>& cards, int* dummies = nullptr) {
  CardMask m = 0;
  for (int c : cards) {
    if (c == 52) { if (dummies) ++*dummies; continue; }
    m |= cugo::core::card_bit(static_cast<CardId>(c - 1));
  }
  return m;
}

std::string cards_str(CardMask m) {
  std::string s;
  while (m != 0) {
    const CardId c = cugo::core::pop_first_card(m);
    if (!s.empty()) s += ',';
    s += std::to_string(c + 1);
  }
  return s;
}

// Splits a SNAP line into its global part and the two player parts.
struct SnapLine {
  std::string global;
  std::string player[2];
};

SnapLine split_snap(const std::string& line) {
  SnapLine s;
  const std::size_t a = line.find(" | p0 ");
  const std::size_t b = line.find(" | p1 ");
  s.global = line.substr(0, a);
  s.player[0] = line.substr(a + 3, b - a - 3);
  s.player[1] = line.substr(b + 3);
  return s;
}

struct Report {
  int games = 0;
  int clean = 0;
  int skipped = 0;
  std::map<std::string, int> first_mismatch;
  int printed = 0;
  int max_print = 20;
};

class Replayer {
 public:
  Replayer(const std::vector<std::string>& lines, int game, Report& report)
      : lines_(lines), game_(game), report_(report) {}

  // Returns the mismatch category ("" when the game replayed identically).
  std::string run() {
    for (index_ = 0; index_ < lines_.size(); ++index_) {
      const std::string& line = lines_[index_];
      std::string err;
      if (line.rfind("DECK ", 0) == 0) err = deal(line);
      else if (line.rfind("MSG MissionStartToC", 0) == 0) err = mission(line);
      else if (line.rfind("ACT ", 0) == 0) err = act(line);
      else if (line.rfind("SNAP ", 0) == 0) err = snap(line, false);
      else if (line.rfind("FINAL ", 0) == 0) err = snap(line, true);
      else if (line.rfind("MSG MoneyChangeToC", 0) == 0) money(line);
      else if (line.rfind("MSG GameResultToC", 0) == 0) err = result(line);
      else if (line.rfind("MSG NoticePushToC", 0) == 0 && num(line, "nReason") == 0) err = nagari();
      else if (line.rfind("ERROR", 0) == 0 || line == "STALL") return "skip:recording";
      if (!err.empty()) return err;
    }
    return "";
  }

  std::string detail;

 private:
  const std::vector<std::string>& lines_;
  int game_;
  Report& report_;
  std::size_t index_ = 0;
  TerminalGameState50 state_{};
  std::vector<CardId> forced_cards_;
  ForcedDraws50 forced_{};
  long long side_expected0_ = 0;
  bool started_ = false;

  std::string fail(const std::string& category, const std::string& what) {
    std::ostringstream os;
    os << "game " << game_ << " line " << index_ << " [" << category << "] " << what
       << "\n    at: " << lines_[index_].substr(0, 200);
    detail = os.str();
    return category;
  }

  std::string deal(const std::string& line) {
    const int first = static_cast<int>(num(line, "first"));
    const std::vector<int> order = ints(field(line, "order"));
    if (order.size() != 50) return fail("trace", "deck size");
    std::vector<int> table(order.begin(), order.begin() + 8);
    std::vector<int> hand0(order.begin() + 8, order.begin() + 18);
    std::vector<int> hand1(order.begin() + 18, order.begin() + 28);
    InitialDeal50 deal{};
    deal.floor = mask_of(table);
    deal.hand0 = mask_of(hand0);
    deal.hand1 = mask_of(hand1);
    deal.stock = cugo::core::kShinMatgoDeckMask & ~(deal.floor | deal.hand0 | deal.hand1);
    deal.rng_state = 0x5eed;
    forced_cards_.clear();
    for (std::size_t i = 28; i < order.size(); ++i)
      forced_cards_.push_back(static_cast<CardId>(order[i] - 1));
    forced_ = ForcedDraws50{forced_cards_.data(), static_cast<int>(forced_cards_.size()), 0, false};
    g_forced_draws50 = &forced_;
    state_ = make_terminal_game_state50(deal, static_cast<std::uint8_t>(first), 1, false);
    started_ = true;
    return "";
  }

  std::string mission(const std::string& line) {
    const int type = static_cast<int>(num(line, "nType"));
    const int mult = static_cast<int>(num(line, "nMultiplier"));
    state_.mission_type = static_cast<std::uint8_t>(type);
    state_.mission_multiplier = static_cast<std::uint8_t>(type == kMissionTriple50 ? 0 : mult);
    state_.special.game.turn.mission_cards = mask_of(ints(field(line, "listMissionCard")));
    if (type == kMissionNoCondition50) {
      state_.mission_multi0 = static_cast<std::uint8_t>(mult);
      state_.mission_multi1 = static_cast<std::uint8_t>(mult);
    }
    if (type == kMissionTriple50) {
      update_missions50(state_, 0, 0);
      update_missions50(state_, 1, 0);
    }
    return "";
  }

  // Decisions recorded right after a THROW. A shake prompt answer means the THROW only
  // asked the question and the play itself is recorded again as the next THROW; otherwise
  // the following floor choices belong to this play. Returns the shake answer (-1: none).
  int follow_ups(std::vector<int>& selects) {
    for (std::size_t j = index_ + 1; j < lines_.size(); ++j) {
      const std::string& l = lines_[j];
      if (l.rfind("ACT ", 0) != 0) continue;
      std::istringstream in(l.substr(4));
      int who; std::string kind; int v;
      in >> who >> kind >> v;
      if (kind == "SHAKE") return selects.empty() ? (v != 0 ? 1 : 0) : -1;
      if (kind == "SELECT") { selects.push_back(v); continue; }
      break;
    }
    return -1;
  }
  int pending_shake_ = -1;

 public:
  int pi_aligned_ = 0;

 private:

  ActionResult50 apply(Action50 action) {
    const int saved = forced_.next;
    const ActionResult50 r = apply_action50(state_, action);
    if (r.status != ActionStatus50::kOk) forced_.next = saved;
    return r;
  }

  std::string act(const std::string& line) {
    std::istringstream in(line.substr(4));
    int who; std::string kind; int v = 0;
    in >> who >> kind >> v;
    if (kind == "DealStartToS" || kind == "DealEndToS" || kind == "SHAKE" || kind == "SELECT") return "";
    if (!started_) return fail("trace", "action before deal");
    const int player = policy_decision_player50(state_);
    const PolicyDecision50 decision = policy_decision_kind50(state_);
    auto expect = [&](PolicyDecision50 d, const char* name) -> std::string {
      if (decision != d) {
        std::ostringstream os;
        os << "HG " << name << " but Cugo decision kind " << static_cast<int>(decision)
           << " for player " << player;
        return fail(std::string("flow:") + name, os.str());
      }
      if (player != who) return fail("actor", "HG actor " + std::to_string(who) + " Cugo " + std::to_string(player));
      return "";
    };

    if (kind == "GO") {
      if (auto e = expect(PolicyDecision50::kGoStop, "go"); !e.empty()) return e;
      const auto r = apply(make_action50(v ? ActionKind50::kGo : ActionKind50::kStop));
      return r.status == ActionStatus50::kOk ? "" : fail("apply:go", "status " + std::to_string(static_cast<int>(r.status)));
    }
    if (kind == "GUKJIN") {
      if (auto e = expect(PolicyDecision50::kGukjin, "gukjin"); !e.empty()) return e;
      const auto r = apply(make_action50(v ? ActionKind50::kGukjinPi : ActionKind50::kGukjinAnimal));
      return r.status == ActionStatus50::kOk ? "" : fail("apply:gukjin", "status " + std::to_string(static_cast<int>(r.status)));
    }
    if (kind == "CHONGTONG") {
      if (auto e = expect(PolicyDecision50::kChongtong, "chongtong"); !e.empty()) return e;
      const auto r = apply(make_action50(v ? ActionKind50::kChongtongContinue : ActionKind50::kChongtongWin));
      return r.status == ActionStatus50::kOk ? "" : fail("apply:chongtong", "status " + std::to_string(static_cast<int>(r.status)));
    }
    if (kind != "THROW") return fail("trace", "unknown action " + kind);
    if (auto e = expect(PolicyDecision50::kPrimary, "throw"); !e.empty()) return e;

    std::vector<int> selects;
    const int shake_answer = follow_ups(selects);
    if (shake_answer >= 0) {
      pending_shake_ = shake_answer;
      return "";
    }
    const bool shake = pending_shake_ == 1;
    pending_shake_ = -1;
    Action50 action = kInvalidAction50;
    if (v == 52) {
      action = make_action50(ActionKind50::kBombCredit);
    } else if (v == 49 || v == 50) {
      action = make_action50(ActionKind50::kPlayBonus);
      action.card = static_cast<CardId>(v - 1);
    } else {
      const CardId card = static_cast<CardId>(v - 1);
      const std::uint8_t month = cugo::core::card_month(card);
      const std::uint16_t bit = static_cast<std::uint16_t>(1u << month);
      if ((legal_bomb_months50(state_.special) & bit) != 0) {
        action = make_action50(ActionKind50::kBomb);
        action.month = month;
      } else if ((legal_grenade_months50(state_.special) & bit) != 0) {
        action = make_action50(ActionKind50::kGrenade);
        action.month = month;
      } else {
        action = make_action50(shake ? ActionKind50::kShake : ActionKind50::kPlayCard);
        action.card = card;
        action.month = month;
      }
    }

    std::vector<Resolve50Choices> candidates;
    const CardId none = cugo::core::kInvalidCard;
    auto id = [](int c) { return static_cast<CardId>(c - 1); };
    if (selects.empty()) candidates.push_back(kNoResolve50Choices);
    else if (selects.size() == 1) {
      candidates.push_back(Resolve50Choices{id(selects[0]), none});
      candidates.push_back(Resolve50Choices{none, id(selects[0])});
    } else {
      candidates.push_back(Resolve50Choices{id(selects[0]), id(selects[1])});
      candidates.push_back(Resolve50Choices{id(selects[1]), id(selects[0])});
    }
    ActionResult50 r{};
    for (const auto& c : candidates) {
      action.resolve_choices = c;
      r = apply(action);
      if (r.status == ActionStatus50::kOk) break;
    }
    if (r.status != ActionStatus50::kOk) {
      std::ostringstream os;
      os << "kind " << static_cast<int>(action.kind) << " card " << v << " selects " << selects.size()
         << " status " << static_cast<int>(r.status);
      return fail("apply:throw:" + std::to_string(static_cast<int>(r.status)), os.str());
    }
    if (forced_.mismatch) return fail("draw", "forced draw not in stock");
    return "";
  }

  void money(const std::string& line) {
    if (num(line, "nReason") == 0) return;  // the final settlement
    const long long amount = num(line, "lChangeMoney") / 1000000;
    side_expected0_ += num(line, "nWinnerIdx") == 0 ? amount : -amount;
  }

  std::string snap(const std::string& line, bool final) {
    if (!started_) return "";
    const SnapLine s = split_snap(line);
    const TurnState50& t = state_.special.game.turn;
    if (!final && !terminal50_is_finished(state_)) {
      const int cur = static_cast<int>(num(s.global, "cur"));
      if (t.actor != cur) return fail("turn:actor", "HG cur " + std::to_string(cur) + " Cugo " + std::to_string(t.actor));
    }
    // Equal-value pi outside the mission are interchangeable for scoring, and the two plain
    // pi of one month are interchangeable everywhere. Hangame picks among them by collection
    // order, which Cugo does not track, so align the piles when they differ only that way.
    {
      TurnState50& tm = state_.special.game.turn;
      const CardMask m = tm.mission_cards;
      // Classes of interchangeable cards: same pi value and same mission membership.
      const CardMask doubles = kFixedDoublePiMask | cugo::core::card_bit(cugo::core::kBonusTwoPi);
      const CardMask classes[4] = {kPlainPiMask & ~m, kPlainPiMask & m, doubles & ~m, doubles & m};
      const CardMask any = kPlainPiMask | doubles;
      const CardMask hgf = mask_of(ints(field(s.global, "table")));
      const CardMask hg0 = mask_of(ints(field(s.player[0], "cap")));
      const CardMask hg1 = mask_of(ints(field(s.player[1], "cap")));
      auto same_pile = [&](CardMask a, CardMask b) {
        if ((a & ~any) != (b & ~any)) return false;
        for (CardMask c : classes)
          if (cugo::core::card_count(a & c) != cugo::core::card_count(b & c)) return false;
        return true;
      };
      // On the floor the month matters too; a month's two plain pi share mission membership.
      auto same_floor = [&](CardMask a, CardMask b) {
        if ((a & ~kPlainPiMask) != (b & ~kPlainPiMask)) return false;
        for (std::uint8_t month = 0; month < cugo::core::kMonthCount; ++month) {
          const CardMask mm = cugo::core::month_mask(month) & kPlainPiMask;
          if (cugo::core::card_count(a & mm) != cugo::core::card_count(b & mm)) return false;
        }
        return true;
      };
      const bool differs = hgf != tm.floor || hg0 != tm.captured0 || hg1 != tm.captured1;
      if (differs && same_floor(hgf, tm.floor) && same_pile(hg0, tm.captured0) &&
          same_pile(hg1, tm.captured1) && (hgf | hg0 | hg1) == (tm.floor | tm.captured0 | tm.captured1)) {
        tm.floor = hgf;
        tm.captured0 = hg0;
        tm.captured1 = hg1;
        ++pi_aligned_;
      }
    }
    const CardMask floor = mask_of(ints(field(s.global, "table")));
    if (floor != t.floor)
      return fail("floor", "HG " + cards_str(floor) + " Cugo " + cards_str(t.floor));
    for (int p = 0; p < 2; ++p) {
      const std::string& ps = s.player[p];
      int dummies = 0;
      const CardMask hand = mask_of(ints(field(ps, "hand")), &dummies);
      const CardMask cugo_hand = p == 0 ? t.hand0 : t.hand1;
      if (hand != cugo_hand)
        return fail("hand", "p" + std::to_string(p) + " HG " + cards_str(hand) + " Cugo " + cards_str(cugo_hand));
      if (dummies != credits50(state_.special, static_cast<std::uint8_t>(p)))
        return fail("bomb-dummy", "p" + std::to_string(p) + " HG " + std::to_string(dummies) + " Cugo " +
                                      std::to_string(credits50(state_.special, static_cast<std::uint8_t>(p))));
      const CardMask cap = mask_of(ints(field(ps, "cap")));
      const CardMask cugo_cap = p == 0 ? t.captured0 : t.captured1;
      if (cap != cugo_cap)
        return fail("captured", "p" + std::to_string(p) + " HG " + cards_str(cap) + " Cugo " + cards_str(cugo_cap));
      const ScoreBreakdown score = persistent_score_player50(t, static_cast<std::uint8_t>(p));
      const int hg_point = static_cast<int>(num(ps, "nPoint"));
      // Hangame's running score includes the Go count.
      const int cugo_point = score.total_points + go_count_for_player50(state_.special.game, static_cast<std::uint8_t>(p));
      // A fixed-score end (Chongtong, three-ppuk) pays no pi-bak; Hangame still settles the
      // loser's gukjin as pi there, which changes nothing, so the running score is skipped.
      const bool fixed_end = final && state_.terminal_reason != TerminalReason50::kNone &&
                             state_.terminal_reason != TerminalReason50::kStop;
      if (hg_point != cugo_point && !fixed_end)
        return fail("score", "p" + std::to_string(p) + " HG " + std::to_string(hg_point) + " Cugo " + std::to_string(cugo_point) +
                                 " (pi HG " + field(ps, "nPeeCnt") + " Cugo " + std::to_string(score.pi_units) + ")");
      const int hg_go = static_cast<int>(num(ps, "m_nGoCnt"));
      const int cugo_go = go_count_for_player50(state_.special.game, static_cast<std::uint8_t>(p));
      if (hg_go != cugo_go) return fail("go", "p" + std::to_string(p) + " HG " + std::to_string(hg_go) + " Cugo " + std::to_string(cugo_go));
      static const char* kPuk[] = {"NONE", "FIRST_PUK", "SECOND_PUK", "THIRD_PUK", "ONE_PUK", "TWO_PUK", "THREE_PUK"};
      const std::string hg_puk = field(ps, "m_PukState");
      const int cugo_puk = ppuk_streak50(state_, static_cast<std::uint8_t>(p));
      if (cugo_puk > 6 || hg_puk != kPuk[cugo_puk])
        return fail("ppuk-state", "p" + std::to_string(p) + " HG " + hg_puk + " Cugo " + std::to_string(cugo_puk));
      const int hg_shake = static_cast<int>(num(ps, "_nShakeCnt"));
      const int cugo_shake = shake_count50(state_.special, static_cast<std::uint8_t>(p)) +
                             bombs50(state_.special, static_cast<std::uint8_t>(p));
      if (hg_shake != cugo_shake)
        return fail("shake", "p" + std::to_string(p) + " HG " + std::to_string(hg_shake) + " Cugo " + std::to_string(cugo_shake));
      const int hg_mission = static_cast<int>(num(ps, "m_nMissionMulti"));
      const int cugo_mission = mission_multi50(state_, static_cast<std::uint8_t>(p));
      if (hg_mission != cugo_mission && !(hg_mission <= 1 && cugo_mission <= 1))
        return fail("mission", "p" + std::to_string(p) + " HG " + std::to_string(hg_mission) + " Cugo " + std::to_string(cugo_mission));
    }
    return "";
  }

  // Hangame signals a drawn game (nagari) with a push notice of reason 0 and deals again.
  std::string nagari() {
    if (terminal50_is_finished(state_) || terminal50_has_chongtong_choice(state_) ||
        game50_has_pending_choice(state_.special.game) || !rollout_hands_exhausted50(state_))
      return fail("result:nagari", "HG nagari, Cugo decision kind " +
                                       std::to_string(static_cast<int>(policy_decision_kind50(state_))));
    return "";
  }

  std::string result(const std::string& line) {
    if (!terminal50_is_finished(state_))
      return fail("result:unfinished", "HG ended, Cugo decision kind " +
                                           std::to_string(static_cast<int>(policy_decision_kind50(state_))));
    const Settlement50 s = settle_terminal50(state_);
    const int hg_winner = static_cast<int>(num(line, "nWinnerIdx"));
    const long long hg_score = num(line, "nScore");
    const std::string history = field(line, "szHistory");
    if (hg_winner != s.winner)
      return fail("result:winner", "HG " + std::to_string(hg_winner) + " Cugo " + std::to_string(s.winner));
    if (static_cast<unsigned long long>(hg_score) != s.final_points) {
      std::ostringstream os;
      os << "HG " << hg_score << " Cugo " << s.final_points << " (base " << int(s.winner_base_points)
         << " go " << int(s.winner_go_count) << " gomul " << s.go_multiplier << " bomb/shake " << s.bomb_shake_multiplier
         << " bak " << s.bak_multiplier << " mission " << s.mission_multiplier << " flags " << s.flags
         << " reason " << int(s.reason) << ") HG history " << history;
      return fail("result:points", os.str());
    }
    if (side_expected0_ != state_.side_points0)
      return fail("result:side", "HG " + std::to_string(side_expected0_) + " Cugo " + std::to_string(state_.side_points0));
    return "";
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: cugo_hg_replay50 <trace.txt> [max_reports]\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  Report report;
  if (argc > 2) report.max_print = std::stoi(argv[2]);
  std::vector<std::string> game;
  int index = -1;
  auto flush = [&]() {
    if (index < 0) return;
    ++report.games;
    Replayer r(game, index, report);
    const std::string outcome = r.run();
    if (outcome.empty()) ++report.clean;
    else if (outcome.rfind("skip:", 0) == 0) ++report.skipped, ++report.first_mismatch[outcome];
    else {
      ++report.first_mismatch[outcome];
      if (report.printed++ < report.max_print) std::cout << r.detail << "\n";
    }
    game.clear();
  };
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.rfind("GAME ", 0) == 0) {
      flush();
      index = std::stoi(line.substr(5));
      continue;
    }
    game.push_back(line);
  }
  flush();
  g_forced_draws50 = nullptr;
  std::cout << "games=" << report.games << " identical=" << report.clean << " skipped=" << report.skipped << "\n";
  for (const auto& [k, v] : report.first_mismatch) std::cout << "  " << k << ": " << v << "\n";
  return report.clean + report.skipped == report.games ? 0 : 1;
}
