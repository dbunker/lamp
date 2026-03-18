d2(o3).
d1(o0).
d0(o0).
d1(o9).
d0(o4).
d0(o9).
d0(o8).
d0(o1).
d4(o6).

d4(X) :- d2(X), d1(X).
d2(X) :- d3(X), d5(X).
d4(X) :- d0(X), d1(X).
d2(X) :- d0(X), d1(X).
d2(X) :- d0(X), d1(X).
d3(X) :- d0(X), d4(X).
d2(X) :- d3(X), d5(X).
d4(X) :- d3(X), d5(X).
d0(X) :- d2(X), d4(X).
d5(X) :- d3(X), d4(X).
d5(X) :- d3(X), d4(X).