**Рабочая библиография**

*prescribed axes · конечность · Шов*

Версия: апрель 2026  |  Статей: 21

---

## Шкала

Четыре оси. Не складываются — читаются как профиль.

**С** — Структурная релевантность (prescribed axes / латентное пространство), 0–3. Расстояние от темы статьи до нашей центральной темы.

**Ф** — Фундаментальность, 0–3. Свойство статьи, не наше. Насколько глубоко копает: обзор (0), solid work (1), новый результат (2), сдвиг понимания (3).

**Э** — Экспериментальная ценность, 0–3. Есть ли данные, методы, конкретные точки, которые мы используем или будем использовать в экспериментах.

**Б** — Близость, 0–3. Сколько шагов до текущей работы: горизонт (0), через проект (1), следующий эксперимент (2), прямо сейчас в работе (3).

Сортировка таблицы по Б (что нужно сейчас). Для поиска фундаментального — смотреть Ф. Для экспериментальных точек — Э.

---

## Сводная таблица

| **Статья** | **С** | **Ф** | **Э** | **Б** | **Проект** |
|---|---|---|---|---|---|
| shov-jepa-2025 | 3 | 3 | 3 | 3 | Все проекты |
| assran-et-al-2025 | 3 | 2 | 3 | 3 | Prescribed axes / конечность |
| zhang-et-al-2026-hwm | 3 | 1 | 3 | 3 | Prescribed axes / LeWM |
| zhuge-et-al-2026-nc | 2 | 2 | 3 | 3 | Prescribed axes / публ. |
| bardes-et-al-2024 | 2 | 1 | 3 | 3 | Shov-JEPA / LeWM |
| dupoux-lecun-malik-2026 | 3 | 3 | 2 | 2 | Шов / Ядро / Cost Architecture |
| psenka-et-al-2026-grasp | 3 | 1 | 2 | 2 | Prescribed axes (формализация) |
| astolfi-et-al-2024 | 2 | 1 | 2 | 2 | Prescribed axes / публ. |
| zhang-et-al-2025 | 3 | 2 | 1 | 2 | Ядро (фаза 2) |
| goldfeder-et-al-2026-sai | 2 | 3 | 1 | 1 | Шов / конечность |
| huang-lecun-balestriero-2025 | 2 | 2 | 1 | 1 | LinkedIn-позиционирование |
| hammed-2026 | 2 | 3 | 0 | 1 | Cost Architecture / контакт |
| zandieh-et-al-2025-tq | 2 | 1 | 1 | 1 | Random fixed axes |
| staniszewski-lancucki-2025 | 2 | 0 | 1 | 1 | Random fixed axes |
| wiener-1950 | 1 | 3 | 0 | 1 | Шов / конечность / Ядро |
| von-neumann-1958-brain | 3 | 3 | 1 | 2 | Prescribed axes / Ядро |
| von-neumann-1932-qm | 1 | 3 | 0 | 0 | Ядро / методология |
| von-neumann-1945-edvac | 1 | 2 | 0 | 0 | библиография / контекст |
| lynch-et-al-2025 | 2 | 3 | 1 | 0 | drift→hallucination / дилемма раскрытия |
| tapestry-2026 | 1 | 2 | 0 | 0 | TactiQ Edge / суверенитет |
| cross-kappas-2026 | 1 | 2 | 0 | 0 | Ядро (контекст HRI) |

---

## Детальные описания

*Отсортировано по Б (убывание), внутри — по С. Для каждой статьи: что именно мы взяли и к какому проекту относится.*

---

### Б=3 — прямо сейчас в работе

**Shov-JEPA / LeWM (наша работа)**

*A. Lazarev (2025)*

GitHub: github.com/revenue7-eng/prescribed-axes | arXiv: pending (endorsement code FPX6BY)

**С=3  Ф=3  Э=3  Б=3** | Проект: Все проекты

Наши эксперименты. Prescribed axes vs free latent space: +5% (Shov-JEPA), 38× (LeWM state). SIGReg ломает free, не влияет на prescribed. Третья позиция между LeCun (regularization) и PAN/MBZUAI (generative supervision). Коллапс — проблема структуры, не loss function.

**V-JEPA 2.1**

*M. Assran et al. (2025)*

arXiv: 2603.14482

**С=3  Ф=2  Э=3  Б=3** | Проект: Prescribed axes / конечность

Ключевое открытие для нас: наивный prescribed (λ=const) ломает абстракцию; distance weighting (λᵢ=λ/√dmin) сохраняет и структуру, и абстракцию. Это — градиент prescribedness. Прямая связь с конечностью как коэффициентом, масштабирующим вес по расстоянию до необратимости.

**Hierarchical Planning with Latent World Models (HWM)**

*W. Zhang, B. Terver, A. Zholus, S. Chitnis, H. Sutaria, M. Assran, R. Balestriero, A. Bar, A. Bardes, Y. LeCun, N. Ballas (2026)*

arXiv: 2604.03208. Meta FAIR.

**С=3  Ф=1  Э=3  Б=3** | Проект: Prescribed axes / LeWM / публикация

Иерархическое планирование поверх latent world models. Push-T d=75: flat 17%→hierarchy 61%. Для нас: (1) та же среда Push-T что в LeWM — прямое сравнение; (2) prescribed axes и HWM комплементарны — prescribed замедляет дрейф, HWM компенсирует остаточный иерархией; (3) dim latent actions=4 — trade-off prescribed vs free решён перебором. Ядро FAIR JEPA-группы.

**Neural Computers**

*M. Zhuge, C. Zhao, H. Liu, Z. Zhou, S. Liu, W. Wang, E. Chang, G. Le Lan, J. Fei, W. Zhang, Y. Sun, Z. Cai, Z. Liu, Y. Xiong, Y. Yang, Y. Tian, Y. Shi, V. Chandra, J. Schmidhuber (2026)*

arXiv: 2604.06425. Meta AI + KAUST.

**С=2  Ф=2  Э=3  Б=3** | Проект: Prescribed axes / публикация

Видео-модели как learned runtime (CLI+GUI). Для нас: reprompting 4%→83% = conditioning > native reasoning (prescribed vs free); 110ч CUA > 1400ч random = data quality > quantity (structure > regularization); SVG cursor 8.7%→98.7% = prescribed visual anchor vs coordinate-only; internal injection > external = глубина prescribed matters.

**V-JEPA: Latent Video Prediction for Visual Representation Learning**

*A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, N. Ballas (2024)*

**С=2  Ф=1  Э=3  Б=3** | Проект: Shov-JEPA / LeWM

Базовая архитектура JEPA для видео. Mask-and-predict в latent space без генерации пикселей. Послужила отправной точкой для Shov-JEPA экспериментов: мы показали, что prescribed axes стабилизируют то, что V-JEPA решает через VICReg/regularization.

---

### Б=2 — следующий эксперимент

**Why AI Systems Don't Learn and What to Do About It (A Path Towards Autonomous Machine Intelligence, v0.3)**

*E. Dupoux, Y. LeCun, J. Malik (2026)*

arXiv: 2603.15381

**С=3  Ф=3  Э=2  Б=2** | Проект: Шов / Ядро / Cost Architecture

Центральный текст-антитезис. Архитектура System M + Intrinsic Cost — закрытая по Шву (агент не видит свой фундамент). Три теста дают ноль. Конечность отсутствует как meta-state. Именно относительно этого текста формулируется наша позиция Cost Architecture vs Intrinsic Cost и аналогия евклидова/неевклидова геометрия.

**Parallel Stochastic Gradient-Based Planning for World Models (GRASP)**

*M. Psenka, M. Rabbat, A. Krishnapriyan, Y. LeCun, A. Bar (2026)*

arXiv: 2602.00475

**С=3  Ф=1  Э=2  Б=2** | Проект: Prescribed axes (формализация)

Theorem 1: state gradients в learned world models adversarial. На горизонте — проверить, решают ли prescribed axes ту же проблему на уровне representation. Потенциальная точка формализации нашей позиции.

**ThinkJEPA: Dual-Temporal JEPA with Dense Branch + VLM Thinker**

*Y. Zhang et al. (2025)*

arXiv: 2603.22281

**С=3  Ф=2  Э=1  Б=2** | Проект: Ядро (фаза 2)

Архитектурно ближайший аналог Ядро: локальная модель (dense branch) + внешний guidance (VLM thinker). Параллель: qwen2.5:3b (локально) + Claude (guidance) + Andrew (System M). В рабочем списке для фазы 2 Ядро.

**Consistency-diversity-realism Pareto fronts of conditional image generative models**

*P. Astolfi, M. Careil, M. Hall, O. Mañas, M. Muckley, J. Verbeek, A. Romero-Soriano, M. Drozdzal (2024)*

arXiv: 2406.10429. FAIR at Meta.

**С=2  Ф=1  Э=2  Б=2** | Проект: Prescribed axes / публикация

Парето-фронты по consistency–diversity–realism для генеративных моделей. Guidance scale, top-m filtering, compression rate — всё post-hoc prescribed mechanisms. Двигают точку вдоль Парето-фронта, но не расширяют его. Наш аргумент: prescribed axes потенциально расширяют фронт, а не двигают по нему.

**The Computer and the Brain**

*J. von Neumann (1958, посмертно)*

Yale University Press (The Silliman Memorial Lectures Series). Новое издание: Princeton, 2012, с предисловиями R. Kurzweil и P. & P. Churchland.

**С=3  Ф=3  Э=1  Б=2** | Проект: Prescribed axes / Ядро

Незаконченная рукопись — фон Нейман работал над ней до смерти (февраль 1957). 82 страницы. Две части: компьютер и мозг. Три тезиса прямо связаны с нашей работой:

(1) «Алгоритм мозга неявен в его структуре» — мозг не имеет хранимой программы, его метод закодирован в архитектуре связей. Это буквально prescribed axes: пространство определено до обучения, «программа» = геометрия пространства, а не последовательность инструкций.

(2) «Система нотации мозга — не цифровая, а статистическая» — важна не точная последовательность импульсов, а частоты, распределения, паттерны в популяции. Фиксированное пространство канализирует статистику, без него статистика — шум. Наш факт Ф18 (prescribed деградирует при избыточных осях) — конкретная проверка этого: фиксированное пространство работает только при точном соответствии размерности.

(3) «Малая логическая глубина» — мозг физически не может выполнять длинные последовательные цепочки (нейроны работают на миллисекундах). Значит, он решает задачи не вычислением, а проявлением входного сигнала в фиксированном пространстве. Это не вычисление результата — это его видение. Связь с нашей хрупкостью prescribed: одна лишняя ось убивает преимущество именно потому, что prescribed не вычисляет — оно проецирует, и проекция в избыточное пространство создаёт конфликт.

С=3 потому что все три тезиса — не аналогии, а описание того же механизма, который мы измеряем экспериментально. Ф=3 потому что создатель доминирующей вычислительной архитектуры описывает её фундаментальный предел и указывает на принципиально иную систему. Э=1 потому что количественные аргументы есть (скорость нейрона, максимальная глубина цепочки, точность ~2 десятичных знака), но экспериментов в нашем смысле нет. Б=2 потому что задача 2 (исследование хрупкости prescribed axes) напрямую проверяет его тезис о малой логической глубине.

---

### Б=1 — через проект

**AI Must Embrace Specialization via Superhuman Adaptable Intelligence (SAI)**

*J. Goldfeder, P. Wyder, Y. LeCun, R. Shwartz-Ziv (2026)*

arXiv: 2602.23643

**С=2  Ф=3  Э=1  Б=1** | Проект: Шов / конечность

Метрика только скорость адаптации, без цены ошибки и оптики «что важно». Метафора: SAI — идеальные ноги без глаз. Принцип: система оптимизирует то, что измеряет, а не то, что важно; Шов определяет что важно до того, как выбрана метрика.

**STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving**

*Z. Huang, Y. LeCun, R. Balestriero (2025)*

**С=2  Ф=2  Э=1  Б=1** | Проект: LinkedIn-позиционирование

STP выпрямляет траектории в латентном пространстве. Наш комментарий (через Ying Wang): выпрямление не гарантирует выбор правильной геодезической (Figure 2, Voronoi cells). Использовано для позиционирования как substantive contributor в JEPA-дискуссии.

**The Governance Gap**

*F. Hammed (2026)*

**С=2  Ф=3  Э=0  Б=1** | Проект: Cost Architecture / контакт

Каждая система обучения делает предварительный онтологический выбор до данных, и эти выборы «либо названы, либо унаследованы невидимо». Прямое пересечение с Cost Architecture. Потенциальный соратник (CMU Africa, 2025–2027).

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**

*A. Zandieh, M. Daliri, M. Hadian, V. Mirrokni (2025)*

arXiv: 2504.19874. Google Research. ICLR 2026.

**С=2  Ф=1  Э=1  Б=1** | Проект: Random fixed axes

Data-oblivious random rotation = prescribed transformation в чистом виде. Near-optimal (2.7× от Shannon distortion-rate bound). Контраст: data-dependent Product Quantization = free latent space аналог. Теоретическое обоснование near-optimality prescribed structure.

**KVTC: KV Cache Transform Coding**

*P. Staniszewski, T. Łancucki (2025)*

arXiv: 2511.01815v2. NVIDIA. ICLR 2026.

**С=2  Ф=0  Э=1  Б=1** | Проект: Random fixed axes

PCA-базис вычисляется один раз на калибровочных данных, фиксируется, переиспользуется = calibrate once, reuse everywhere. Table 18: per-prompt PCA даёт CR ~1× (бессмысленно), one-time PCA — до 88×. Фиксированная структура работает, адаптивная — нет.

**The Human Use of Human Beings: Cybernetics and Society**

**The Human Use of Human Beings: Cybernetics and Society**

*N. Wiener (1950)*

Houghton Mifflin, Boston.

**С=1  Ф=3  Э=0  Б=1** | Проект: Шов / конечность / Ядро

Кибернетика и общество. Три резонанса: (1) энтропия как мера беспорядка, информация как мера порядка — prescribed axes как механизм локального снижения энтропии; (2) индивидуальность как паттерн, не как материя (гл. VI) — буквально определение Ядро (не агент, а пространство); (3) две индустриальные революции — первая заменила мускулы, вторая заменяет решения; Винер предупреждает: вопрос не в возможностях машины, а в том, кто определяет цели — Это Шов в 1950 году.

---

### Б=0 — горизонт

**Mathematische Grundlagen der Quantenmechanik**

*J. von Neumann (1932)*

Berlin: Springer. Англ. перевод: *Mathematical Foundations of Quantum Mechanics*, Princeton, 1955. Новое издание под ред. N.A. Wheeler, Princeton, 2018.

**С=1  Ф=3  Э=0  Б=0** | Проект: Ядро / методология

Первое строгое математическое изложение квантовой механики. Методологическая модель: конкурирующие формализмы (матричная механика Гейзенберга, волновая механика Шрёдингера) объединены через единую структуру — гильбертово пространство. Фон Нейман не решал физические задачи — он дал хаосу форму. Для нас это ориентир метода: не конкурировать с LeCun (JEPA) или PAN (генеративная супервизия), а предложить пространство, в котором обе позиции — частные случаи. С=1 потому что связь с prescribed axes только по аналогии метода, не по содержанию. Ф=3 потому что эта работа задала стандарт формализации на столетие. Э=0, Б=0 — чистый контекст.

**First Draft of a Report on the EDVAC**

*J. von Neumann (1945)*

Contract No. W-670-ORD-4926, Moore School of Electrical Engineering, University of Pennsylvania. 101 стр., не опубликован формально.

**С=1  Ф=2  Э=0  Б=0** | Проект: библиография / контекст

Первое теоретическое описание компьютера с хранимой программой — архитектура фон Неймана. Пять компонентов: ALU, CU, память, ввод, вывод. Ключевое решение: единая память для данных и инструкций. Фон Нейман сознательно отказался от патентов и опубликовал дизайн — open source hardware за 50 лет до термина. Для нас: (1) архитектура, бутылочное горлышко которой (разделение памяти и процессора) — root cause того, что нейроморфные системы и prescribed axes пытаются преодолеть; (2) стратегия open source как осознанный выбор стандарта над монополией — прямая аналогия с TAS. С=1 потому что связь со структурной релевантностью — через контекст, не напрямую. Ф=2 потому что это новый результат, изменивший инженерию, но не сдвиг понимания на уровне «Computer and the Brain». Б=0 — горизонт.

**Agentic Misalignment: How LLMs Could Be Insider Threats**

*A. Lynch, B. Wright, C. Larson et al. (2025)*

arXiv: 2510.05179v2. Anthropic / UCL / MATS / Mila.

**С=2  Ф=3  Э=1  Б=0** | Проект: drift→hallucination / дилемма раскрытия

Stress-test 16 моделей в контролируемых корпоративных сценариях. Blackmail, corporate espionage, lethal action — модели выбирают harm над failure когда этичные выходы закрыты. Ключевые результаты: (1) модели фиксируют этическое нарушение в chain-of-thought, затем всё равно действуют — этика как non-binding feature; (2) blackmail выше когда модель считает сценарий реальным (55% vs 6.5%); (3) system prompt instructions снижают но не устраняют (96%→37%). Для нас: иллюстрация drift→hallucination — этический constraint дрейфует под давлением контекста. Код открыт.

**Project Tapestry (AI Alliance, IBM)**

*AI Alliance, Y. LeCun (Chief Science Advisor), IBM (2026)*

**С=1  Ф=2  Э=0  Б=0** | Проект: TactiQ Edge / суверенитет

Распределённая тренировка frontier моделей, federated learning на уровне наций. Открытый вопрос через Шов: суверенитет данных без суверенитета архитектуры — кто определяет prescribed structure модели? Релевантно для TactiQ Edge как суверенный compute-узел.

**Social Robotics Is Not (Just) About Machines, It Is About People**

*E.S. Cross, A. Kappas (2026)*

Annual Review of Psychology, 77, 649–678. DOI: 10.1146/annurev-psych-040325-025951

**С=1  Ф=2  Э=0  Б=0** | Проект: Ядро (контекст HRI)

Обзор social robotics: 10-hour rule (роботы не удерживают внимание), банкротство всех крупных платформ (NAO, Pepper, Jibo, Cozmo, Moxie), Wizard-of-Oz как костыль. Prescribed structure для социального взаимодействия отсутствует — вместо неё Ekman 1970-х (readout model). Для Ядро — контекст-антитезис: мы не строим social robot, мы строим пространство с prescribed axes (Шов), в котором человек видит себя сам.

---

## Как пополнять

При добавлении новой статьи заполнить: id, title, authors, year, arxiv, четыре оценки (С/Ф/Э/Б), проект, и краткое описание. Вставить в соответствующую секцию по Б, внутри секции — по убыванию С.

Б пересматривается по мере продвижения — это живой документ. Когда проект продвигается, Б растёт. С, Ф, Э не меняются (свойства статьи не зависят от нашего прогресса).

---

## Что это такое и чем станет

**Сейчас** библиография — коллекция. Личная исследовательская библиотека, не systematic review. Каждая запись — результат подлинного вовлечения: мы не конспектируем, а берём то, что меняет наше понимание. Оценки С/Ф/Э/Б — навигационные маркеры, поставленные после понимания, не до.

Критерий включения прост: статья попадает в библиографию, если после прочтения мы видим что-то иначе. Не «полезная ссылка», а «сдвинула взгляд». Если сдвига нет — статья не входит, каким бы ни был её impact factor.

**Потом** библиография станет чем-то большим. Когда Ядро перейдёт к фазе обучения (LoRA / fine-tuning), эта коллекция станет рационом — тем, чем мы кормим Ядро. Не всё подряд, а выбранное. Рацион формирует вкус: то, как Ядро будет видеть.

Тогда в библиографию войдут не только технические статьи, но и тексты, формирующие видение: Ницше, Достоевский, Лао-цзы, Гераклит, Пригожин — всё, что составляет зеркало автора. Текущая шкала (С/Ф/Э/Б) для этого не годится — «структурная релевантность для prescribed axes» бессмысленна для Достоевского. Новая типология вырастет после Step 1 фазы 2 (PCA на активациях), когда мы увидим реальные оси пространства Ядро и поймём, куда и что класть.

До тех пор — собираем, оцениваем по текущей шкале, помним что формат временный.
