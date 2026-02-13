## الخلاصة
تعتمد **نماذج تحويل** السلاسل التقليدية على الشبكات العصبية التكرارية المعقدة أو شبكات طي تتضمن مُرمِّز وفاك ترميز.
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.

تربط النماذج الأعلى أداءً هذين المكوّنين عبر آلية الانتباه.
The best performing models also connect the encoder and decoder through an attention mechanism. 

يقترح الباحثون في هذه الورقة بنية شبكية جديدة وبسيطة، تُسمّى **المحوّل**، تعتمد كليًا على آليات الانتباه دون أي حاجة إلى التكرار أو الطي (الالتفاف).
We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

أظهرت التجربة على مهمّتَي ترجمة آلية تفوّق هذا النموذج من حيث جودة المخرجات، وقابليته للمعالجة المتوازية، وتطلّبه زمن تدريب أقل.
Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

وقد حقّق المحول نتيجة بلغت 28.4 على **مقياس التقييم ثنائي اللغة** في مهمّة الترجمة من الإنجليزية إلى الألمانية ضمن مجموعة بيانات **ورشة عمل الترجمة الآلية لعام 2014**، متجاوزًا أفضل النتائج السابقة، بما في ذلك نتائج **النماذج المجمّعة**، بفارق نقطتين على مقياس التقييم المذكور. وحقّق المحوّل في مهمّة الترجمة من الإنجليزية إلى الفرنسية على مجموعة البيانات ذاتها نتيجة متقدمة لنموذج منفرد بلغت 41.8، وذلك بعد تدريب استمر 3.5 أيام فقط على ثماني وحدات معالجة رسومية. 
Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.

تُعدّ كلفة الحوسبة تلك جزءًا يسيرًا مقارنةً بتكاليف تدريب أفضل النماذج الحالية. ووضح الباحثون أن نموذج المحول يمكن تعميمه بشكل جيد على مهام أخرى، إذ جرى تطبيقه بنجاح في تحليل التكوين اللغوي للغة الإنجليزية باستخدام بيانات تدريب واسعة وأخرى ومحدودة.
On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

## المصطلحات التأسيسية
<abbr title="تعمل على تحويل سلاسل الإدخال إلى سلاسل الإخراج من خلال بنيتي المُرمِّز وفاك الترميز، وتستخدم عادةً في أنظمة الترجمة الآلية وغيرها من المهام ذات الطبيعة المتسلسلة.">نماذج تحويل</abbr>
<abbr title="BELU يُستخدم لقياس جودة الترجمة الآلية عبر مقارنة عدد من الكلمات n-gram المتطابقة بين مخرجات النموذج والترجمات المرجعية (واحدة أو أكثر)، مع فرض عقوبة تقلّل الدرجة في حال كانت الترجمة الناتجة قصيرة بصورة مبالغ فيها.">مقياس التقييم ثنائي اللغة</abbr>
<abbr title="قدمت مجموعة بيانات ترجمة مرجعية مستخدمة على نطاق واسع لتقييم نماذج الترجمة الآلية، وتحتوي على ملايين أزواج الجمل المتوافقة بين الإنقليزية والألمانية، و الإنقليزية والفرنسية.">ورشة عمل الترجمة الآلية لعام 2014</abbr><abbr title="ensembles تُشير إلى استخدام عدة نماذج تُدمَج مخرجاتها معًا بهدف تحقيق أداء يفوق ما يمكن أن يقدّمه نموذج منفرد.">النماذج المجمّعة</abbr>
<abbr title="RNN هي نماذج شبكية تُعالج السلاسل بشكل تدريجي خطوةً بعد أخرى، مع الاحتفاظ بحالة داخلية تعتمد على المخرجات السابقة، الأمر الذي يجعل إمكانية الحوسبة المتوازية فيها محدودة أو غير متاحة.">الشبكات العصبية التكرارية</abbr>
<abbr title="LSTM نوع من الشبكات العصبية التكرارية تُستخدم على نطاق واسع في نمذجة السلاسل تحويلها من شكل إلى آخر مثل الترجمة الآلية، وهي احد اشكال الشبكات التكرارية التي يهدف المحول إلى استبدالها.">شبكات الذاكرة طويلة-قصيرة المدى</abbr>
<abbr title="GRU هي نوع من الشبكات التكرارية يعتمد على آليات البوابات لتنظيم تدفّق المعلومات، ما يجعلها أكثر كفاءة في تعلّم السلاسل مقارنةً بالشبكات التكرارية ذات البنية الأساسية.">الشبكات التكرارية ذات البوابات</abbr>
<abbr title="تتألف من جزئين رئيسين؛ المُرمِّز الذي يحوّل سلاسل الدخل إلى تمثيلات متتابعة، وفاكّ الترميز الذي يستخدم هذه التمثيلات لتوليد سلاسل خرج النموذج خطوةً بخطوة. تُعدّ هذه البنية صيغة معيارية في مهام تحويل السلاسل مثل الترجمة الآلية، وتمثّل لبّ المحوّل.">الترميز وفكه</abbr>
<abbr title="آلية تحدد أي من رموز سلاسل الدخل أو الخرج يجب على النموذج التركيز عليها.">الإنتباه</abbr>

## المقدمة
أصبحت **الشبكات العصبية التكرارية**، ولا سيما **شبكات الذاكرة طويلة-قصيرة المدى**[^12] و**الشبكات التكرارية ذات البوابات**[^7]، من أحدث الأساليب في نمذجة السلاسل ومسائل التحويل، مثل نمذجة اللغة والترجمة الآلية [^29] [^2] [^5].
Recurrent neural networks, long short-term memory [12] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [29, 2, 5].

ومنذ ذلك الحين، تتواصل الجهود لتوسيع آفاق نماذج اللغة المتكررة وهياكل **الترميز وفكه**[^31] [^21] [^13].
Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [31, 21, 13].

<br />

تقسّم النماذج التكرارية عمليات الحوسبة تبعاً لمواقع الرموز في سلاسل الدخل والخرج.
Recurrent models typically factor computation along the symbol positions of the input and output sequences.

وبمواءمة تلك المواقع مع خطوات الحوسبة المتعاقبة، تُنتج هذه النماذج تسلسلاً من الحالات المخفية $h_t$ كدالة تعتمد على كلٍ من الحالة المخفية السابقة $h_{t-1}$، ومدخلات الموقع $t$. 
Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht 1 and the input for position t.

تُعيق تلك الطبيعة المتسلسلة المتأصّلة إمكانية إجراء حوسبة متوازية لعينات التدريب، وهو أمر بالغ الأهمية عند التعامل مع السلاسل الطويلة، حيث تفرض قيود الذاكرة حدودًا على حجم الدفعات الممكن معالجتها.
This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. 

وقد حسّنت الدراسات الحديثة من كفاءة الحوسبة بشكل ملحوظ عبر حيل التقسيم [^18] والحوسبة المشروطة [^26]، مع تسجيل مكاسب في الأداء في الحالة الأخيرة.
Recent work has achieved significant improvements in computational efficiency through factorization tricks [18] and conditional computation [26], while also improving model performance in case of the latter.

ويبقى مع ذلك القيد الجوهري للحوسبة المتسلسلة قائماً.
The fundamental constraint of sequential computation, however, remains.

<br />

أصبحت آليات **الانتباه** جزء لا يتجزء من نماذج سلاسل البيانات ونماذج التحويل عبر طيف واسع من المهام، إذ تسمح بنمذجة الاعتماديات بين الرموز بغضّ النظر عن بُعدها في سلاسل الدخل أو الخرج [^2] [^16].
Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 16].

ومع ذلك تُستخدم آليّة الإنتباه في جميع الحالات تقريباً [^22] بوصفها مكملاً للشبكات العصبية التكرارية وليس بديلاً عنها.
In all but a few cases [22], however, such attention mechanisms are used in conjunction with a recurrent network.

<br />

يقترح الباحثون في هذه الورقة نموذج المحول بوصفه بنية تتخلّى بالكامل عن الشبكات التكرارية، وتعتمد كلياً على آليات الانتباه لاستخلاص العلاقات الشمولية بين سلاسل الدخل وسلاسل الخرج.
In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. 

يُتيح نموذج المحول درجة عالية جداً من الحوسبة المتوازية، ما يمكنه من تحقيق مستوى جديد من الجودة في الترجمة بعد تدريب لا يتجاوز اثنتي عشرة ساعة على ثماني وحدات معالجة رسومية.
The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.


## الخلفية
شكل هدف تقليل العمليات الحسابية التسلسلية أساسًا للنماذج وحدة المعالجة الرسومية العصبية الموسعة [20]، وشبكة بايت [15]، وطي سلسلة لسلسلة [8]، تستخدم تلك النماذج شبكات الطي العصبية كحجر بناء أساسي، حيث تحسب التمثيلات المخفية بالتوازي لجميع مواضع الدخل والخرج.
The goal of reducing sequential computation also forms the foundation of the Extended Neural Graphical Processing Unit [20], ByteNet [15] and ConvS2S [8], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions.

يزداد عدد العمليات اللازمة لربط الإشارات في تلك النماذج من موضعَي دخل أو خرج عشوائيين مع ازدياد المسافة بينهما، خطيًا في نموذج طي سلسلة لسلسلة ولوغاريتميًا في شبكة بايت.
In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet.

ما يجعل تعلم العلاقات بين المواضع البعيدة أكثر صعوبة [11].
This makes it more difficult to learn dependencies between distant positions [11]. 

يُختزل عدد العمليات الحسابية في المحول إلى عدد ثابت، وإن كان ذلك على حساب انخفاض الدقة الفعالة نتيجةً لحساب متوسط ​​المواضع موزوناً بالانتباه، وهو تأثير عالجه الباحثون باستخدام آلية الانتباه متعدد الرؤوس كما هو موضح في القسم 3.2.
In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

<br />

الانتباه الذاتي، يُسمى أحيانًا بالانتباه الداخلي، هو آلية انتباه تربط بين المواضع المختلفة لسلسلة واحدة بهدف حساب تمثيل لهذه السلسلة.
Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.

وقد استُخدم الانتباه الذاتي بنجاح في مجموعة متنوعة من المهام، بما في ذلك فهم القراءة، والتلخيص التجريدي، والاستلزام النصي، وتعلم تمثيلات الجمل المستقلة عن المهمة [4، 22، 23، 19].
Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 22, 23, 19].


## المصدر
https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

## المراجع
[^2]: Dzmitry Bahdanau , Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.
[^5]: Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.
[^7]: Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.
[^12]: Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.
[^13]: Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.
[^16]: Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations, 2017.
[^18]: Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722, 2017.
[^21]: Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attentionbased neural machine translation. arXiv preprint arXiv:1508.04025, 2015.
[^22]: Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016.
[^26]: Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.
[^29]: Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104–3112, 2014.
[^31]: Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.
[^32]: Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.