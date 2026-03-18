d3(o1).
d2(o3).
d1(o5).
d5(o0).
d4(o0).
d2(o0).
d3(o0).
d4(o2).
d5(o3).

d2(X) :- d0(X), d5(X), d1(X).
d0(X) :- d2(X), d3(X), d4(X).
d1(X) :- d2(X), d5(X), d4(X).
d0(X) :- d3(X), d5(X), d1(X).
d3(X) :- d5(X), d1(X), d4(X).
d5(X) :- d2(X), d1(X), d4(X).
d2(X) :- d0(X), d1(X), d4(X).
d2(X) :- d0(X), d5(X), d4(X).
d4(X) :- d0(X), d5(X), d1(X).
d4(X) :- d3(X), d5(X), d1(X).
d2(X) :- d3(X), d1(X), d4(X).