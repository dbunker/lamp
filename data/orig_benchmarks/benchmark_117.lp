d3(o2).
d1(o5).
d1(o1).
d0(o4).
d3(o0).

d2(X) :- not d1(X), d5(X).
d2(X) :- d4(X), not d5(X).
d5(X) :- d2(X), not d3(X).
d0(X) :- d4(X), not d2(X).
d4(X) :- d2(X), not d1(X).
d0(X) :- d1(X), not d4(X).
d4(X) :- d2(X), not d0(X).
d1(X) :- d0(X), not d5(X).
d1(X) :- d4(X), not d3(X).
d2(X) :- not d3(X), d5(X).
d0(X) :- not d5(X), d3(X).