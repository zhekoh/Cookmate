// Quiz component. Markup:
// <div class="quiz" data-answer="1" data-why="one line shown after answering">
//   <p class="q">Question?</p>
//   <div class="opts"><button>Option A</button><button>Option B</button></div>
//   <p class="fb"></p>
// </div>
// Feedback is immediate. Options should be the same length so formatting gives no hint.
// A .score element (if present) shows correct / total once every quiz on the page is answered.
(function () {
  const quizzes = Array.from(document.querySelectorAll(".quiz"));
  const results = new Map();
  function render() {
    const s = document.querySelector(".score");
    if (!s || results.size < quizzes.length) return;
    const right = Array.from(results.values()).filter(Boolean).length;
    s.textContent = `Retrieval score: ${right} / ${quizzes.length}` + (right === quizzes.length ? " — all correct." : " — reread the ones marked, then come back tomorrow and retake.");
  }
  quizzes.forEach((quiz, idx) => {
    const answer = Number(quiz.dataset.answer);
    const why = quiz.dataset.why || "";
    const buttons = Array.from(quiz.querySelectorAll(".opts button"));
    const fb = quiz.querySelector(".fb");
    buttons.forEach((btn, i) => {
      btn.addEventListener("click", () => {
        if (results.has(idx)) return;
        const ok = i === answer;
        results.set(idx, ok);
        btn.classList.add(ok ? "ok" : "bad");
        buttons[answer].classList.add("ok");
        buttons.forEach((b) => (b.disabled = true));
        fb.className = "fb " + (ok ? "ok" : "bad");
        fb.textContent = (ok ? "Right. " : "Not quite. ") + why;
        render();
      });
    });
  });
})();
