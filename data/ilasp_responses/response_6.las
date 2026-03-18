%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 0.761s
%% Conflict analysis                       : 19.027s
%%   - Positive Examples                   : 19.027s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 6.674s
%% Total                                   : 26.795s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d1(V1) :- d4(V1).
d8(V1) :- d1(V1).
d7(V1) :- d8(V1).
d6(V1) :- d5(V1).
d7(V1) :- d3(V1).
d3(V1) :- d6(V1); not d8(V1).
d8(V1) :- d2(V1); not d5(V1).

