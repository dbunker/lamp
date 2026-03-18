%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 1.498s
%% Conflict analysis                       : 19.042s
%%   - Positive Examples                   : 19.042s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 7.557s
%% Total                                   : 28.988s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d4(V1) :- d3(V1).
d7(V1) :- d3(V1).
d1(V1) :- d3(V1).
d5(V1) :- d0(V1).
d4(V1) :- d0(V1).
d8(V1) :- d6(V1).
d2(V1) :- d4(V1); not d0(V1).

