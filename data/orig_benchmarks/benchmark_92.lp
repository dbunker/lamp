d3(o1).
d1(o5).
d5(o1).
d1(o4).
d1(o0).
d1(o1).
d5(o5).
d2(o0).
d2(o1).

d0(X) :- d1(X), d5(X), not d4(X).
d3(X) :- d1(X), d0(X), not d5(X).
d1(X) :- d4(X), d0(X), not d2(X).
d0(X) :- d1(X), not d5(X), d3(X).
d3(X) :- d2(X), d0(X), not d1(X).
d4(X) :- d2(X), d1(X), not d3(X).
d3(X) :- d0(X), d5(X), not d2(X).
d2(X) :- d4(X), d0(X), not d1(X).
d3(X) :- d4(X), d0(X), not d2(X).
d1(X) :- d4(X), d0(X), not d3(X).
d5(X) :- d1(X), d3(X), not d2(X).