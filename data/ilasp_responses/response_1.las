%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 1.289s
%% Conflict analysis                       : 38.485s
%%   - Positive Examples                   : 38.485s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 11.223s
%% Total                                   : 51.573s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d8(V1) :- d2(V1).
d5(V1) :- d0(V1).
d6(V1) :- d7(V1).
d1(V1) :- d4(V1).
d2(V1) :- d5(V1); not d1(V1).
d8(V1) :- d1(V1); not d5(V1).

