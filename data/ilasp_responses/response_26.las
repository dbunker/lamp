%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 1.354s
%% Conflict analysis                       : 45.365s
%%   - Positive Examples                   : 45.365s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 15.423s
%% Total                                   : 62.978s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d1(V1) :- d8(V1).
d2(V1) :- d7(V1).
d3(V1) :- d7(V1).
d8(V1) :- d7(V1).
d9(V1) :- d5(V1).
d3(V1) :- d4(V1).
d0(V1) :- d2(V1); not d7(V1).

