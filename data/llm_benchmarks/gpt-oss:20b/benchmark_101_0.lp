d4(o0).
d2(o1).
d0(o5).
d2(o0).
d5(o4).
d3(o0).

d4(X) :- d0(X).
d4(X) :- d5(X).
d4(X) :- d2(X).
d0(X) :- d5(X).
d0(X) :- d2(X), not d3(X).