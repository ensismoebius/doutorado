# Thesis profile fingerprints before and after `degradation.profile` (2026-11-15)

**Problem.** A result file carries `config_hash`, the hash of every declared key of the
configuration that produced it. When the `profile` key became a declared key of every
profile with `degradation.enabled = true` (commit 5671d1a22), the hash of those profiles
changed although nothing they compute did: a reader holding an old `config_hash` cannot find
its profile. This page maps old to new.

**What changed, in numbers.** 466 profiles are listed in `profiles/profiles.manifest`;
64 of them enable the degradation and changed hash; the other 402 do not declare
`profile` and kept theirs. (The `smoke/` mirrors, not in the manifest, changed in the same
way: 64 more files.)

```
old declared config:  ... "degradation": {"enabled": true, "level": 0.25, "scope": "test", "seed": 7}
new declared config:  ... "degradation": {"enabled": true, "level": 0.25, "profile": "whisper_mix", "scope": "test", "seed": 7}
hash = std::hash<std::string>(declared_json().dump())      -> a different 64-bit number
```

**How the values were computed (measured vs reasoned).** With the code of 5671d1a22, each
profile was parsed and its `declared_json()` taken, `experiment.run_tag` extended with
`_rep0` as `thesis.cpp` does for repeat 0, and hashed with `std::hash<std::string>` (the
expression of `config_fingerprint`); the old value is the same JSON without
`degradation.profile`. Repeat `r` has the seed raised by `r` and the tag `_rep<r>`, so its
hashes differ from these; the map is for repeat 0. **Not verified against a run:** the
committed Phase 01 summaries were written by an older configuration schema (no
`dataset.source`, another `dataset.root`), so their `config_hash` cannot be matched to either
column; the hashes are also `std::hash`, i.e. specific to libstdc++.

| profile | old `config_hash` (rep 0) | new `config_hash` (rep 0) |
|---|---|---|
| `biomex/bx01_dysph_earlypron_probes_l100.json` | 14525589032402263705 | 6764046066790265001 |
| `biomex/bx01_dysph_earlypron_probes_l25.json` | 10722704617656962971 | 1720334889019329509 |
| `biomex/bx01_dysph_earlypron_probes_l50.json` | 8929054829323579133 | 10374772411884406295 |
| `biomex/bx01_dysph_earlypron_probes_l75.json` | 10176740971606969993 | 2947998615003361986 |
| `biomex/bx01_dysph_earlypron_test_l100.json` | 12250940589199415550 | 11107619488450711286 |
| `biomex/bx01_dysph_earlypron_test_l25.json` | 14746145971107674258 | 8386079882189318468 |
| `biomex/bx01_dysph_earlypron_test_l50.json` | 6670644457127544824 | 15663405685524973329 |
| `biomex/bx01_dysph_earlypron_test_l75.json` | 13434475606700965242 | 10375936078634730878 |
| `biomex/bx01_dysph_latepron_probes_l100.json` | 18192023666588345718 | 14681248643641927911 |
| `biomex/bx01_dysph_latepron_probes_l25.json` | 7182498715634861996 | 7282605916548410828 |
| `biomex/bx01_dysph_latepron_probes_l50.json` | 8856338501268950887 | 6762280043149663307 |
| `biomex/bx01_dysph_latepron_probes_l75.json` | 9047114730908366488 | 9279110502226787156 |
| `biomex/bx01_dysph_latepron_test_l100.json` | 5585866817878677975 | 17110788207443280299 |
| `biomex/bx01_dysph_latepron_test_l25.json` | 3486003814115863382 | 7093934568987905090 |
| `biomex/bx01_dysph_latepron_test_l50.json` | 17673635904252262199 | 11556305394016562121 |
| `biomex/bx01_dysph_latepron_test_l75.json` | 17740661333357668887 | 2759438204751550693 |
| `biomex/bx01_dysph_voice_probes_l100.json` | 9788534238197919578 | 11053101818410221943 |
| `biomex/bx01_dysph_voice_probes_l25.json` | 6364458882851605510 | 12659194404201509617 |
| `biomex/bx01_dysph_voice_probes_l50.json` | 12028779404497720525 | 18385392955096003479 |
| `biomex/bx01_dysph_voice_probes_l75.json` | 16442772955697740227 | 12842612680210600271 |
| `biomex/bx01_dysph_voice_test_l100.json` | 13068169746133763100 | 14968319192876068974 |
| `biomex/bx01_dysph_voice_test_l25.json` | 13192400378228791587 | 10203565866877054500 |
| `biomex/bx01_dysph_voice_test_l50.json` | 13340111591498537261 | 5991735005881128178 |
| `biomex/bx01_dysph_voice_test_l75.json` | 15503462293727232893 | 14491788311409690570 |
| `phase01/p01_dysph_earlyimag_probes_l100.json` | 15072874250364030369 | 9823760354615043612 |
| `phase01/p01_dysph_earlyimag_probes_l25.json` | 4918732355902436725 | 17234778091356264837 |
| `phase01/p01_dysph_earlyimag_probes_l50.json` | 4198573208563373913 | 17594183338335053253 |
| `phase01/p01_dysph_earlyimag_probes_l75.json` | 1509787214448072208 | 13190684955487357668 |
| `phase01/p01_dysph_earlyimag_test_l100.json` | 14393015753117085835 | 6850104362890046076 |
| `phase01/p01_dysph_earlyimag_test_l25.json` | 6763161147137994555 | 3684446231673885047 |
| `phase01/p01_dysph_earlyimag_test_l50.json` | 5178307227049477652 | 1884814012878624442 |
| `phase01/p01_dysph_earlyimag_test_l75.json` | 13620915213615714121 | 8807214843405339489 |
| `phase01/p01_dysph_earlypron_probes_l100.json` | 2422464126654166029 | 6195454840266126293 |
| `phase01/p01_dysph_earlypron_probes_l25.json` | 8747057272672552486 | 8652331668613950654 |
| `phase01/p01_dysph_earlypron_probes_l50.json` | 1353833468229674085 | 17424033228806139238 |
| `phase01/p01_dysph_earlypron_probes_l75.json` | 3584798699181389150 | 5338046888857227072 |
| `phase01/p01_dysph_earlypron_test_l100.json` | 2942971327964348512 | 16061019673057842751 |
| `phase01/p01_dysph_earlypron_test_l25.json` | 5177345609878821806 | 18004315991168838208 |
| `phase01/p01_dysph_earlypron_test_l50.json` | 17694591539005505462 | 781936591396459661 |
| `phase01/p01_dysph_earlypron_test_l75.json` | 9148070198001260806 | 17411267531303588667 |
| `phase01/p01_dysph_lateimag_probes_l100.json` | 15412150796107578277 | 10699036395394917250 |
| `phase01/p01_dysph_lateimag_probes_l25.json` | 7561613500030314125 | 15369134782460550544 |
| `phase01/p01_dysph_lateimag_probes_l50.json` | 14381581261430265980 | 2646297454396086318 |
| `phase01/p01_dysph_lateimag_probes_l75.json` | 10898338542705585245 | 16809464650904985890 |
| `phase01/p01_dysph_lateimag_test_l100.json` | 15121638068166589897 | 11692748278517206764 |
| `phase01/p01_dysph_lateimag_test_l25.json` | 16183203044164653062 | 17327326283378999507 |
| `phase01/p01_dysph_lateimag_test_l50.json` | 10111661838783120147 | 1522500143274806325 |
| `phase01/p01_dysph_lateimag_test_l75.json` | 8524910906765186184 | 895673190192274469 |
| `phase01/p01_dysph_latepron_probes_l100.json` | 7812495045992889203 | 5732759753131749390 |
| `phase01/p01_dysph_latepron_probes_l25.json` | 1039728421580950263 | 17775105285039350739 |
| `phase01/p01_dysph_latepron_probes_l50.json` | 14139789893135060264 | 11669629004420464888 |
| `phase01/p01_dysph_latepron_probes_l75.json` | 10849478126310468370 | 5180415357061596266 |
| `phase01/p01_dysph_latepron_test_l100.json` | 2504543712068730789 | 151902538523890089 |
| `phase01/p01_dysph_latepron_test_l25.json` | 805514409841679845 | 1489617341072579139 |
| `phase01/p01_dysph_latepron_test_l50.json` | 381604967684659465 | 9106862622410137960 |
| `phase01/p01_dysph_latepron_test_l75.json` | 12123608332546281200 | 15215960395826871253 |
| `phase01/p01_dysph_voice_probes_l100.json` | 8871919533023138357 | 10205701772657754669 |
| `phase01/p01_dysph_voice_probes_l25.json` | 8357093904759380280 | 13758600066502913619 |
| `phase01/p01_dysph_voice_probes_l50.json` | 1236529280527506974 | 1425538845383442340 |
| `phase01/p01_dysph_voice_probes_l75.json` | 7664705763517205844 | 9016467033567692249 |
| `phase01/p01_dysph_voice_test_l100.json` | 17981016106729706292 | 4367829764834528822 |
| `phase01/p01_dysph_voice_test_l25.json` | 2039711611436569960 | 12045848049243344673 |
| `phase01/p01_dysph_voice_test_l50.json` | 9289722841612537240 | 14324758048284425209 |
| `phase01/p01_dysph_voice_test_l75.json` | 16817828988177639523 | 6482920902381887170 |
